import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder


class Dataset:
    def __init__(self, df, config):
        self.config = config
        self.datetime = self.config.field_config["datetime"]
        self.df = self.preprocess(df)
        self.train_df = self.gen_train_df(self.df)
        self.predict_df = self.gen_predict_df(self.df)
        self.dataset = self.gen_dataset()
        self.train_dataloader, self.val_dataloader = self.gen_dataloader()

    def fill_na(self, df):
        for c in self.config.field_config["need_fill_na"]:
            df[c] = df[c].fillna(
                df.groupby(self.config.dataset_config["group_ids"])[c].transform("max")
            )
        return df

    def preprocess(self, df):
        df[self.datetime] = pd.to_datetime(df[self.datetime])
        for c in self.config.field_config["category"]:
            df[c] = df[c].astype(str).astype("category")

        df["time_idx"] = (
            df.groupby(self.config.dataset_config["group_ids"])[self.datetime]
            .rank(method="dense")
            .astype(int)
            - 1
        )
        df = self.fill_na(df)
        return df

    def gen_train_df(self, df):
        df = df.dropna(subset=[self.config.dataset_config["target"]]).loc[:]
        df = df.reset_index(drop=True)
        return df

    def gen_predict_df(self, df):
        encoder_data = df.dropna(subset=[self.config.dataset_config["target"]]).loc[:]
        encoder_data = encoder_data[
            lambda x: x.time_idx
            > x.time_idx.max() - self.config.dataset_config["max_encoder_length"]
        ]
        decoder_data = df[df[self.config.dataset_config["target"]].isnull()]
        decoder_data[self.config.dataset_config["target"]] = decoder_data[
            self.config.dataset_config["target"]
        ].fillna(value=0)
        df = pd.concat([encoder_data, decoder_data], ignore_index=True)
        return df

    def gen_dataset(self):
        training_cutoff = (
            self.train_df["time_idx"].max()
            - self.config.dataset_config["max_prediction_length"]
        )
        if not isinstance(self.config.dataset_config["target"], list):
            target_normalizer = GroupNormalizer(
                groups=self.config.dataset_config["group_ids"],
                transformation="softplus",
            )
        else:
            target_normalizer = MultiNormalizer(
                normalizers=[
                    GroupNormalizer(
                        groups=self.config.dataset_config["group_ids"],
                        transformation="softplus",
                    ),
                    GroupNormalizer(
                        groups=self.config.dataset_config["group_ids"],
                        transformation="softplus",
                    ),
                ]
            )
        target_normalizer = None
        categorical_encoders = [NaNLabelEncoder(add_nan=True)] * len(
            self.config.field_config["need_encode_na"]
        )
        categorical_encoders = dict(
            zip(self.config.field_config["need_encode_na"], categorical_encoders)
        )
        if self.config.valid:
            dataset = TimeSeriesDataSet(
                self.train_df[lambda x: x.time_idx <= training_cutoff],
                target_normalizer=target_normalizer,
                categorical_encoders=categorical_encoders,
                **self.config.dataset_config
            )
        else:
            dataset = TimeSeriesDataSet(
                self.train_df,
                target_normalizer=target_normalizer,
                categorical_encoders=categorical_encoders,
                **self.config.dataset_config
            )
        return dataset

    def gen_dataloader(self):
        train_dataloader = self.dataset.to_dataloader(
            train=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        if self.config.valid:
            validation = TimeSeriesDataSet.from_dataset(
                self.dataset, self.train_df, predict=True, stop_randomization=True
            )
            val_dataloader = validation.to_dataloader(
                train=False,
                batch_size=self.config.batch_size * 10,
                num_workers=self.config.num_workers,
            )
            return train_dataloader, val_dataloader
        else:
            return train_dataloader, None

