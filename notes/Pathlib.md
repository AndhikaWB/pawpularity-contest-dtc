

`pathlib` should be used instead of `os.path` when possible. It's especially useful for managing compatibility between Windows system (e.g. local host) and Linux system (e.g. cloud environment), which `os.path.join` can't handle elegantly.

1. Use `pathlib` only when working with local path, not network path (e.g. S3 path)
    - `pathlib` will replace // with / (e.g. from `s3://example/path` to `s3:/example/path`), making it incorrect
    - `pathlib` will also normalize `./path/file` into `path/file`, which may not be acceptable for some CLI programs
2. If you're using `pathlib` inside a function, the function input (string) shouldn't be redeclared as `pathlib` object directly
    - This will cause problem if that input is returned later as a `pathlib` object, and the one who read it expects a string (e.g. `boto3` library)
    - Also, a string object can be appended via the `+` operator, but `pathlib` will raise an error instead (not sure why they made this decision)
    - This makes `pathlib` naturally incompatible for some operations, like prepending a `pandas` string series with a `pathlib` object (unless the `pathlib` object is converted to string first)

## Extras

Note that `os.path` is not designed for network path either. When converting from Linux to Windows path, you usually call `mypath.replace("/", "\\")`. This will also modify S3 path from `s3://example/path` to `s3:\\example\path`, which is wrong. You can use regex as workaround but I consider it an overkill.

The alternative for handling network path is by using `urllib.parse` (built-in) or the `cloudpathlib` (external) library.