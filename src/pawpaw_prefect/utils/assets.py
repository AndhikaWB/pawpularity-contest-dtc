import re
import jinja2
import inspect
import functools
from pathlib import Path

from typing import Any, Callable, Iterable, Unpack, overload
from prefect.tasks import Task, MaterializingTask, TaskOptions

from prefect.assets import Asset
from prefect.context import AssetContext


class Template(Asset):
    # Prevent key validation on template
    key: str


def validate_templates(
    known_vars: dict[str, Any], templates: Iterable[str | Template] | None,
    extra_vars: dict[str, Any] | None = None
) -> list[Asset]:
    """Validate asset templates with Jinja and ensure that each asset has a valid URI
    (will fallback to `file://` URI if there's no URI).

    Args:
        known_vars (dict[str, Any]): Known variables and their values (e.g. from the
            task input arguments). None values will be filtered.
        templates (Iterable[str | Template] | None): The asset templates that will be
            filled by the variable values. If the type is a `Template`, the "key" will
            be used instead.
        extra_vars (dict[str, Any], optional): Extra variables (e.g. for holding
            fallback values) that can be used in the templates. None values will be
            filtered. Defaults to None.

    Returns:
        list[Asset]: List of validated assets.
    """

    if not templates:
        return []

    if isinstance(templates, str):
        raise ValueError(
            # Don't loop string characters, generating incorrect assets
            f'Asset templates should be in a list or tuple, not string directly'
        )

    # Raise error instead of using empty string on undefined variable
    jinja_env = jinja2.Environment(undefined = jinja2.StrictUndefined)
    # None values are not undefined and still need to be filtered manually
    known_vars = {k: v for k, v in known_vars.items() if v is not None}
    if extra_vars:
        extra_vars = {k: v for k, v in extra_vars.items() if v is not None}

    # Use set to ensure no duplicate asset
    valid_assets: set[Asset] = set()

    for template in templates:
        try:
            asset = jinja_env.from_string(
                template.key if isinstance(template, Template) else template,
                globals = extra_vars
            )

            # Fill the asset template with the real asset value
            asset = asset.render(**known_vars)
        except jinja2.UndefinedError:
            asset = None

        if asset:
            # Check if the asset has a valid URI (e.g. s3://)
            # If not, assume as local file:// and give absolute path
            if not re.search(r'^[a-z0-9]+://', asset):
                asset = Path(asset).resolve().as_uri()

            valid_assets.add(
                Asset.model_validate(
                    dict(template) | {'key': asset} if isinstance(template, Template)
                    else {'key': asset}
                )
            )

    return list(valid_assets)


@overload
def lazy_materialize[**P, R](
    *assets: str | Template, by: str | None = ...,
    asset_deps: list[str | Template] | None = ..., output_as: str | None = ...,
    extra_vars: dict[str, Any] | None = ..., _f: Callable[P, R],
    **task_kwargs: Unpack[TaskOptions]
) -> Callable[P, R]: ...


@overload
def lazy_materialize[**P, R](
    *assets: str | Template, by: str | None = ...,
    asset_deps: list[str | Template] | None = ..., output_as: str | None = ...,
    extra_vars: dict[str, Any] | None = ..., _f: None = ...,
    **task_kwargs: Unpack[TaskOptions]
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def lazy_materialize[**P, R](
    *assets: str | Template, by: str | None = None,
    asset_deps: list[str | Template] | None = None, output_as: str | None = None,
    extra_vars: dict[str, Any] | None = None, _f: Callable[P, R] | None = None,
    **task_kwargs: Unpack[TaskOptions]
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Lazy `@materialize` for Prefect, by taking the asset values from the original
    task input arguments, and/or task output result.

    For example, if your task is `upload_to_s3(local_dir, remote_dir)`, then you can
    simply use the `@lazy_materialize("{{remote_dir}}")` decorator to make `remote_dir`
    as your asset. If `remote_dir` is a nested attribute (e.g. in a Pydantic model), you
    can use `@lazy_materialize("{{remote_dir.target_attr}}")` instead.

    Other examples:
    - Multiple variables: `@lazy_materialize("s3://{{bucket}}/{{remote_dir}}")`
    - Multiple assets: `@lazy_materialize("s3://{{my_bucket}}", "s3://{{other_bucket}}")`
    - Asset dependency: `@lazy_materialize(asset_deps = ["{{local_dir}}"])`
    - Extra variable: `@lazy_materialize("{{fn(dir)}}", extra_vars = {"fn": func})`
    - Task output as asset: `@lazy_materialize("{{output}}", output_as = "output")`

    All variables must be part of the original task input arguments (assets with unknown
    variable will be skipped). If you need extra variables, use the `extra_vars` and/or
    the `output_as` parameter as shown above.

    Args:
        *assets (str | Template, optional): Templates containing variable names (e.g.
            from the task input arguments) that will be used as assets, with the format
            `"{{name}}"` instead of `"name"` directly. If the type is a `Template`, the
            "key" will be used instead.
        by (str, optional): The tool name that materialized the assets. Defaults to
            None.
        asset_deps (list[str | Template], optional): Similar as `assets` but used for
            declaring upstream assets (dependency) rather than downstream assets
            (material). Defaults to None.
        output_as (str, optional): Save the task output result as variable too (if the
            result is not None). You can use any name to refer to it, allowing you to
            use the name in the `assets` templates (but not in `asset_deps`). Defaults
            to None. The output assets will be added late, so some operations (e.g.
            adding metadata) may not work.
        extra_vars (dict[str, Any], optional): Extra variables that are needed in the
            templates, or for holding fallback value(s) in case the original argument(s)
            is missing (undefined). None values will be filtered. Defaults to None.
        **task_kwargs (Any, optional): Other arguments that can be passed to Prefect
            `Task` class.
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Convert all task input arguments as kwargs dict
            known_vars = inspect.getcallargs(f, *args, **kwargs)

            if output_as and output_as in known_vars.keys():
                raise KeyError(f'Variable already used as input: "{output_as}"')

            # Get the real asset and dependency values from the task input arguments
            assets_val = validate_templates(known_vars, assets, extra_vars)
            deps_val = validate_templates(known_vars, asset_deps, extra_vars) or None

            @functools.wraps(f)
            def f_inject(*args: P.args, **kwargs: P.kwargs) -> R:
                output = f(*args, **kwargs)
                
                if output_as and output is not None:
                    # Save the output result as known variable too
                    known_vars[output_as] = output

                    if context := AssetContext.get():
                        # Get current assets directly from the asset context
                        cur_assets = [i.key for i in context.downstream_assets]
                        # Get potential new assets from the task output
                        assets_val = validate_templates(known_vars, assets, extra_vars)

                        # Inject new assets right before ending the task
                        for asset in assets_val:
                            if asset.key not in cur_assets:
                                context.downstream_assets.add(asset)
                        context.update_tracked_assets()

                return output

            prefect_decorator = (
                MaterializingTask(
                    fn = f_inject if output_as else f, assets = assets_val,
                    materialized_by = by, asset_deps = deps_val, **task_kwargs
                ) if assets_val else Task(
                    fn = f_inject if output_as else f, asset_deps = deps_val,
                    **task_kwargs
                )
            )

            return prefect_decorator(*args, **kwargs)
        return wrapper
    return decorator if _f is None else decorator(_f)