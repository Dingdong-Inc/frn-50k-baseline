batch_size = 1024
valid = True
quantiles = 7
num_workers = 8
use_gpu = False
date = None
quantiles_ratio = [0.1, 0.3, 0.5, 0.7, 0.9]


field_config = dict(
    datetime="dt",
    category=[
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        "day_of_week",
        "holiday_flag",
        "activity_flag",
    ],
    need_fill_na=[],
    need_encode_na=[],
)

dataset_config = dict(
    time_idx="time_idx",
    min_prediction_length=3,
    max_prediction_length=7,
    min_encoder_length=21,
    max_encoder_length=42,
    group_ids=["store_id", "product_id"],
    target="sale_amount",
    # weight="",
    static_categoricals=[
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ],
    time_varying_known_categoricals=[
        "day_of_week",
        "holiday_flag",
        "activity_flag",
    ],
    time_varying_known_reals=[
        "time_idx",
        "discount",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["sale_amount"],
    add_relative_time_idx=True,
    extra_decoder=[
    ],
    add_encoder_length=True,
    scalers={
        # 'time_idx': None,
        # 'precpt': None,
        # 'avg_temperature': None,
        # 'avg_humidity': None,
        # 'avg_wind_level': None,
    },
    allow_missing_timesteps=True,
)

model_config = dict(
    learning_rate=0.01,
    hidden_size=32,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=16,
    reduce_on_plateau_patience=4,
    lstm_layers=1
)

trainer_config = dict(
    max_epochs=5,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    default_root_dir=None,
)
