from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CompartmentEntity:
    """Represents a compartment in the tissue."""

    compartment_name: str

    def __repr__(self):
        return f"{self.compartment_name}"

    def __eq__(self, value):
        return self.compartment_name == value.compartment_name


@dataclass(frozen=True)
class CellEntity:
    """Represents a population of a particular cell type."""

    cell_type: str
    compartment: CompartmentEntity | None = None

    def __repr__(self):
        if self.compartment:
            return f"{self.cell_type}@{self.compartment}"
        return self.cell_type


@dataclass
class SpatialMetric:
    metric_name: str
    cell_population_a: CellEntity
    cell_population_b: CellEntity | None = (
        None  # if None, then its univariate metric
    )
    sample_id: str = ""
    parameters: dict = field(default_factory=dict)
    values: Any = None
    directional: bool = False  # if order of cell_a, cell_b matters

    def __repr__(self):
        if self.cell_population_b:
            if self.directional:
                pair = f"{self.cell_population_a} -> {self.cell_population_b}"
            else:
                pair = f"{self.cell_population_a} <-> {self.cell_population_b}"
        else:
            pair = f"{self.cell_population_a}"
        return (
            f"<SpatialMetric {self.metric_name} "
            f"[{pair}] "
            f"sample_id={self.sample_id}>"
        )

    # Validate cases
    def __post_init__(self):
        """
        For the metric, validation should be done to ensure valid comparisons
        are made for the specific metric.
        """


@dataclass
class SpatialMetricResults:
    results: list[SpatialMetric] = field(default_factory=list)
    metric_name: str | None = None

    def __post_init__(self):
        if not self.results:
            return
        metric_name = self.results[0].metric_name
        for m in self.results:
            if m.metric_name != metric_name:
                raise ValueError(
                    f"All results must be of the same type of metric. "
                    f"Got {metric_name} and {m.metric_name}."
                )
        self.metric_name = metric_name  # convenience attr

    def __repr__(self):
        return (
            f"<SpatialMetricCollection {self.metric_name} "
            f"n={len(self.results)} instances>"
        )

    def add(self, metric: SpatialMetric) -> None:
        if self.metric_name is None:
            self.metric_name = metric.metric_name
        else:
            if metric.metric_name != self.metric_name:
                raise ValueError(
                    f"All metrics must be of the same type. "
                    f"Got {metric.metric_name} and {self.metric_name}."
                )
            self.results.append(metric)

    def filter(
        self,
        sample_id: str | list[str] | None = None,
        compartment: CompartmentEntity | list[CompartmentEntity] | None = None,
    ) -> SpatialMetric:
        if isinstance(sample_id, str):
            sample_id = [sample_id]

        if isinstance(compartment, str):
            compartment = [compartment]

        return [
            m
            for m in self.results
            if (sample_id is None or m.sample_id in sample_id)
            and (compartment is None or m.compartment in compartment)
        ]

    def cross_reference(
        self, obs_df: pd.DataFrame, sample_id_key: str, group_key: str
    ):
        """
        Based on sample_id has a common key, cross-reference an obs_df
        to retrieve a subset of results for each group.
        """
        assert sample_id_key in obs_df
        assert group_key in obs_df
        grouped_results = {}
        unique_groups = obs_df[group_key].unique()
        for g in unique_groups:
            sub = obs_df[obs_df[group_key] == g]
            ids = sub[sample_id_key].values
            print(ids)
            # query ids;
            res = self.filter(sample_id=ids)
            grouped_results[g] = res
        return grouped_results

    def to_dataframe(self):
        rows = []
        for m in self.results:
            rows.append(
                {
                    "metric": m.metric_name,
                    "cell_type_a": m.cell_population_a,
                    "cell_type_b": m.cell_population_b,
                    "sample_id": m.sample_id,
                    "parameters": m.parameters,
                    "values": m.values,
                }
            )
        return pd.DataFrame(rows)


# # tests below
# import numpy as np

# tumor_compartment = CompartmentEntity("tumor")
# stroma_compartment = CompartmentEntity("stroma")

# self_compartment = SpatialMetric(
#     metric_name="GCross",
#     cell_population_a=CellEntity("CD8"),
#     cell_population_b=CellEntity("CD68"),
#     sample_id="slide42",
#     values=(np.array([5, 10, 15]), np.array([0.1, 0.5, 0.9])),
# )

# cross_compartment = SpatialMetric(
#     metric_name="GCross",
#     cell_population_a=CellEntity("CD8", tumor_compartment),
#     cell_population_b=CellEntity("CD68", stroma_compartment),
#     sample_id="slide42",
#     values=(np.array([5, 10, 15]), np.array([0.1, 0.5, 0.9])),
# )

# coll = SpatialMetricResults(
#     [
#         SpatialMetric(
#             "co_occurrence",
#             "Tcell",
#             "Bcell",
#             sample_id="s1",
#             values=0.2,
#             directional=False,
#         ),
#         SpatialMetric(
#             "co_occurrence",
#             "Bcell",
#             "Bcell",
#             sample_id="s1",
#             values=0.3,
#             directional=False,
#         ),
#     ]
# )

# coll.filter(sample_id="s1")
# coll.to_dataframe()
# coll.cross_reference(
#     pd.DataFrame([["a", "s1"], ["b", "s2"]], columns=["grp", "sample"]),
#     sample_id_key="sample",
#     group_key="grp",
# )
