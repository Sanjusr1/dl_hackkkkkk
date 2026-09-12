# Architecture

```mermaid
flowchart TD
    A["AIDERv2 Dataset<br/>Flood, Fire, Earthquake, Normal"] --> B["Data Preparation<br/>Resize images and clean folder layout"]

    B --> C["Continual Learning Scenario Builder"]

    C --> D1["Task 0<br/>Flood + Earthquake"]
    C --> D2["Task 1<br/>Fire"]
    C --> D3["Task 2<br/>Normal"]

    D1 --> E["Training Pipeline"]
    D2 --> E
    D3 --> E

    E --> F["ResNet18 Backbone<br/>Feature extractor"]
    F --> G["Growing Cosine Classifier Head<br/>Adds new classes task by task"]

    G --> H1["Finetune<br/>Forgetting baseline"]
    G --> H2["DER++<br/>Replay-based continual learning"]
    G --> H3["Joint<br/>Upper bound"]

    H1 --> I["Evaluation"]
    H2 --> I
    H3 --> I

    I --> J["Metrics<br/>ACC, AvgInc, LA, BWT, Forgetting, Macro-F1"]
    J --> K["Plots<br/>Accuracy Matrix, Retention Curve, Comparison Graph"]
    K --> L["Frontend Dashboard<br/>HTML results page"]
    L --> M["Localhost Demo<br/>http://localhost:8000/frontend/index.html"]
```

## Flow

The project first prepares the AIDERv2 disaster image dataset into a clean
train/validation/test layout. The scenario builder converts the dataset into a
continual learning stream, where the model learns disaster classes task by task.

The model uses a ResNet18 backbone to extract image features and a growing cosine
classifier head to add new classes as they appear. The benchmark compares normal
fine-tuning, DER++ replay-based continual learning, and joint training as an
upper bound.

After every task, the system evaluates how much old disaster knowledge is
retained. The final output is a metrics table, plots, and a dashboard that
summarizes accuracy and forgetting.
