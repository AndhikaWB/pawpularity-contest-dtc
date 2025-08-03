# Eval

## Great Expectations (GX)

Seems to be the older player in this field. It has 10k GitHub stars, while most are still at 2k stars. I checked their docs but their code examples look like this:

```python
gxe.ExpectColumnValuesToBeBetween(
    column="product_rating",
    min_value=1,
    max_value=5,
)

gxe.ExpectColumnQuantileValuesToBeBetween(
    column="purchase_amount",
    quantile_ranges={
        "quantiles": [0.5, 0.9],
        "value_ranges": [[50, 200], [500, 2000]],
    },
)
```

This seems pretty limiting to me. It expected me to know my data range in advance. I don't care about that, I just want to know whether my data changed significantly or not, and send me an alert if needed. Besides, I can write that kind of checks by myself if I want to, no need to rely on GX.

However, the plus point is that they have other aspect of monitoring that may be important for some people:
- Freshness (e.g. data must not be older than a certain date)
- Integrity (e.g. user id must be consistent on both user and balance table)
- Missingness (e.g. proportion of non-null values must be above 90%)
- And so on

What I want currently is more on the model metrics side though (the others can pretty much be guaranteed), and GX doesn't seem to have what I want.

## Soda (soda.io)

You can use their dashboard to configure what kind of monitoring you want. The config will be saved as YAML, which you can also create directly without using their dashboard. It sounds good for non-tech person (no code is required), but probably not for me since my goal is to integrate it with my code directly.

YAML example:

```yaml
dataset: databricks_demo/unity_catalog/demo_sales_operations/regional_sales
filter: |
  order_date >= ${var.start_timestamp}
  AND order_date < ${var.end_timestamp}
variables:
  start_timestamp:
    default: DATE_TRUNC('week', CAST('${soda.NOW}' AS TIMESTAMP))
  end_timestamp:
    default: DATE_TRUNC('week', CAST('${soda.NOW}' AS TIMESTAMP)) + INTERVAL '7 days'
checks:
  - row_count:
  - schema:
columns:
  - name: order_id
    data_type: INTEGER
    checks:
      - missing:
          name: Must not have null values
  - name: customer_id
    data_type: INTEGER
    checks:
      - missing:
          name: Must not have null values
  - name: order_date
    data_type: DATE
    checks:
      - missing:
          name: Must not have null values
      - failed_rows:
          name: Cannot be in the future
          expression: order_date > DATE_TRUNC('day', CAST('${soda.NOW} ' AS TIMESTAMP)) +
            INTERVAL '1 day'
          threshold:
            must_be: 0
```

Also, the things that you can monitor seems pretty similar as Great Expectations (e.g. min-max value, data freshness). It's probably not designed for machine learning purpose, I don't see ML mentioned anywhere on their homepage (same as Great Expectations).

**EDIT:** Turns out they some basic metrics like MAPE, RMSE, etc. and time series based check too (e.g. seasonality), but still feels pretty limited and not convincing enough for me. Further more, [that page](https://docs.soda.io/sodacl-reference/anomaly-detection) mentioned they're deprecating it and they just bought NannyML recently, so I guess they will switch to that eventually.

## Whylogs (and WhyLabs)

From what I've understood, whylogs is the metrics calculator/comparer, and WhyLabs is the dashboard used to show those metrics more conveniently. You can use whylogs without WhyLabs.

It advertises itself as "An open-source data logging library for machine learning models and data pipelines", so there's no doubt it at least cover the basic ML metrics.

The example code looks simple enough:

```
result = why.log(pandas=df_target)
prof_view = result.view()

result_ref = why.log(pandas=df_reference)
prof_view_ref = result_ref.view()

visualization = NotebookProfileVisualizer()
visualization.set_profiles(target_profile_view=prof_view, reference_profile_view=prof_view_ref)

visualization.summary_drift_report()
```

That code will show you an embedded HTML which shows the data distribution (histogram), drift level, and other info for each column on the dataframe. However, I can't find anything about what kind of methods or metrics they used, so I guess you need to calculate them yourself.

I also need to see if they support image data too, and they support it, but the code example only shows up to this point (`log_image`, but what I want to know is `summary_drift_report`):

```
{'image/Brightness.stddev:types/boolean': 0, 'image/Brightness.stddev:types/string': 0, 'image/Brightness.stddev:types/object': 0, 'image/Brightness.stddev:cardinality/est': 1.0, 'image/Brightness.stddev:cardinality/upper_1': 1.000049929250618, 'image/Brightness.stddev:cardinality/lower_1': 1.0, 'image/Brightness.stddev:distribution/mean': 25.719757982712743, 'image/Brightness.stddev:distribution/stddev': 0.0, 'image/Brightness.stddev:distribution/n': 1, 'image/Brightness.stddev:distribution/max': 25.719757982712743, 'image/Brightness.stddev:distribution/min': 25.719757982712743, 'image/Brightness.stddev:distribution/q_01': 25.719757982712743, 'image/Brightness.stddev:distribution/q_05': 25.719757982712743, 'image/Brightness.stddev:distribution/q_10': 25.719757982712743, 'image/Brightness.stddev:distribution/q_25': 25.719757982712743, 'image/Brightness.stddev:distribution/median': 25.719757982712743, 'image/Brightness.stddev:distribution/q_75': 25.719757982712743, 'image/Brightness.stddev:distribution/q_90': 25.719757982712743, 'image/Brightness.stddev:distribution/q_95': 25.719757982712743, 'image/Brightness.stddev:distribution/q_99': 25.719757982712743}
```

Sure it shows a verbose info, but I want a summary of what is actually happening, so what if their brightness is at X level? I just want to know if all these changes are significant or not. I can't find more examples involving image data from their main GitHub page, so I guess my search ends here.

Setting that aside, their dashboard (WhyLabs) design looks pretty nice though, I would say even better than Evidently, and probably has the features that I want (e.g. drift comparison). I don't plan to use their dashboard, but it may be a consideration for some people.

The good:
- Pretty simple, but may be too barebone for some people
- License is Apache 2.0

The bad:
- The documentation is very basic
- Last commit (the main branch) is 6 months ago
- They auto-close stale issue after 2 weeks. That's a big red flag for me, some of the closed issues are a good issue that's just ignored

## NannyML

Their main page looks great, it looks like their homepage is written by actual data scientist rather than the sales people. Few things I liked by reading their posts (blog or homepage):
- They mentioned some of the techniques they used, like DLE and CBPE, which allows to estimate performance on drifted data without having the label column. Others only mentioned metrics without actually explaining how they do/optimize it, which I assume just execute the metric and do simple logging behind the scene
- Explained the concept and best practices like:
  - Why we should use test data as reference instead of train data. As comparison, Evidently didn't explain anything about this, just code example
  - Which kind of data drift to consider. I know that there's univariate and multivariate, but which one is actually more important? Do I actually need both? What's the effect if I choose one over the other?
  - Similar as above but now data drift vs model/performance drift. Is data drift really that important to monitor? Is it fine if the data drifted but the model is still doing well?
  - We have a data/model drift, but then what? They showed an example like how to debug the cause (e.g. which feature likely caused it). For comparison, Evidently has this too on their docs, but it's hard to remember where it's because the good resources are always placed at the bottom of the pages

I don't want to list all the pages that I've read, but you can find them pretty easily from here:
- https://www.nannyml.com/blog/
- https://nannyml.readthedocs.io/en/stable/

Overall I liked every information that they wrote, since I'm a beginner in MLOps too. It feels like my knowledge is actually expanding, not just knowing how to use their tools. That alone is a huge plus point for me. Now don't get me wrong, others like Evidently may have their blogs/docs too, but they're usually mixed with product updates or other things so it's harder to notice through a glance.

Here's the bad news though:
- It's more geared towards traditional ML, they haven't mentioned much about LLM or gen AI so far. I know that I'm not dealing with LLM right now but I know it uses different evaluation method than traditional ML
- It's a bit more advanced, you're expected to know what method you want to use, and what metrics. That's not a problem for me, but may be a consideration for some people. By the way, here's one of the code example (it kinda resembles scikit-learn):
  ```python
  estimator = nml.CBPE(
      problem_type='classification_binary',
      y_pred_proba='predicted_probability',
      y_pred='prediction',
      y_true='employed',
      metrics=['roc_auc'],
      chunk_size=chunk_size,
  )
  estimator = estimator.fit(reference_df)
  estimated_performance = estimator.estimate(analysis_df)

  figure = estimated_performance.plot()
  figure.show()
  ```
- I just learned that it has been recently acquired by Soda (yes, the same Soda that I reviewed earlier), so their future is a bit unclear. Besides, I'm not a fan of the UI first approach that Soda currently uses

## Alibi Detect (by Seldon)

The homepage is clearly written by sales people because I can't find the docs anywhere. Also, by looking at their homepage, it makes me think that Alibi is tightly integrated with other Seldon products. However, after looking a bit more, it seems that Alibi can still be used as a standalone tool.

I had to visit their GitHub, clicking one of the function documentation, and backtrack from there to get the [main docs page](https://docs.seldon.io/projects/alibi-detect/en/stable/) (their [GitHub read me](https://github.com/SeldonIO/alibi-detect) is good for starter too). My impression is that it's a slightly more advanced than NannyML. They list which detection method they supported (e.g. VAE, isolation forest) and you're expected to know which one you want to use.

Code example:
```
cd = MMDDrift(x_ref, backend='pytorch', p_val=.05)
preds = cd.predict(x)
```

What's good:
- Kubernetes native. I see Kubernetes mentioned multiple times here and there
- The supported methods and metrics are quite broad, and support image and text too. You can see this on their GitHub read me table
- Reference to research journals that their methods are based on. They provide code example for each case too

What's neutral:
- The docs are too technical and lengthy, has different vibes compared to NannyML, which I liked. I think they assume everyone is a data scientist
- Has no built-in visualization at all, the code example uses Matplotlib which means you need to do everything manually. Depending on your viewpoint, this can be a plus or a minus

The bad:
- As I said, you're expected to know what method or whatever else you want to use
- It uses a custom license which is similar to AGPL, meaning that you can't use it in production without paying

## Evidently AI

It's already been covered by Zoomcamp, so I purposely looked on Evidently last.

The docs structures are not the best (but still better than some others). Firstly, they don't summarizes their components in a single page (e.g. project, report, dashboard), so you need to read multiple pages to see the relationships. After reading, turns out project and dashboard are actually optional.

They promote [presets](https://docs.evidentlyai.com/metrics/all_presets), by default (with pre-configure metrics, etc.) so I have to find it what each preset does behind the scene. Another slightly annoying thing is that I had to search through multiple pages to find what methods they supported. Turns out it's under [customize data drift](https://docs.evidentlyai.com/metrics/customize_data_drift), separated from common metrics like accuracy, MSE, etc. which are accessible [here](https://docs.evidentlyai.com/metrics/all_metrics).

I decided to test Evidently directly and my opinions are written below. They might sound be a bit harsh, but that's only because I get to test many things directly. Also, the bad things here doesn't necessarily mean the other tools are better.

The good:
- Easy to use, and the presets make things simpler especially if you use their dashboard
- The supported metrics and methods are broad enough, but I think not as broad as Alibi Detect (which has nice checkboxes on what kind of media are supported)
- Supports LLM, though I haven't tested it myself

The bad:
- Docs may be a bit confusing at first, but I guess not that bad
- No direct image support for now, but we can use embedding drift as workaround. Embedding must be imported as dataframe
- Only Pandas dataframe is supported as the input. I get it that our test data are usually not that big, but this is still ridiculous to me
- The result dict doesn't have a standard structure/type hint. Different metrics may use different path and the data types are mixed between Numpy and pure Python. This is a nightmare if you want to upload the result to your own database (e.g. to show to Grafana later)
- In theory, presets are nice, but combined with the previous issue it becomes unreliable since you can't guarantee the structure. In the end, you may want to state the metrics manually to get guaranteed structure

The last 2 points are especially bad for me, and this is the result after their recent major API changes (v0.7.11). I guess it may be even worse than this previously.

# Winner

- Technicality: Alibi Detect (very detailed), NannyML (more casual/general)
- Easyness (subjective): Evidently, but only if you want to use their dashboard. Otherwise, I still prefer NannyML

That's it I guess. There are actually more in this field (e.g. DeepEval), but they all support LLM/generative AI only so I'm too lazy to check them for now.