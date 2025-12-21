from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TableRow:
    method: str
    backbone:  str
    params:  str
    flops: str
    one_shot:  str
    five_shot: str
    is_ours: bool = False


class LatexTableGenerator:

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_main_results_table(
        self,
        rows: list[TableRow],
        caption: str = "Comparison with state-of-the-art methods on MiniImageNet",
        label: str = "tab:main_results",
    ) -> str:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Method & Backbone & Params & FLOPs & 1-shot & 5-shot \\",
            r"\midrule",
        ]

        ours_rows = [r for r in rows if r.is_ours]
        other_rows = [r for r in rows if not r.is_ours]

        for row in other_rows:
            line = (
                f"{row.method} & {row.backbone} & {row. params} & "
                f"{row.flops} & {row.one_shot} & {row.five_shot} \\\\"
            )
            lines.append(line)

        if ours_rows:
            lines.append(r"\midrule")

            for row in ours_rows:
                line = (
                    rf"\textbf{{{row.method}}} & {row.backbone} & "
                    rf"\textbf{{{row.params}}} & \textbf{{{row.flops}}} & "
                    rf"\textbf{{{row. one_shot}}} & \textbf{{{row.five_shot}}} \\"
                )
                lines.append(line)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def generate_ablation_table(
        self,
        configs: list[str],
        components: dict[str, list[bool]],
        one_shot_results: list[str],
        five_shot_results: list[str],
        caption: str = "Ablation study on MiniImageNet",
        label: str = "tab:ablation",
    ) -> str:
        component_names = list(components.keys())
        num_components = len(component_names)

        header_components = " & ".join(component_names)

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{l" + "c" * num_components + "cc}",
            r"\toprule",
            f"Config & {header_components} & 1-shot & 5-shot \\\\",
            r"\midrule",
        ]

        for idx, config in enumerate(configs):
            component_values = []

            for comp_name in component_names: 
                value = components[comp_name][idx]
                component_values.append(r"\checkmark" if value else "-")

            comp_str = " & ".join(component_values)
            line = f"{config} & {comp_str} & {one_shot_results[idx]} & {five_shot_results[idx]} \\\\"
            lines.append(line)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n". join(lines)

    def generate_efficiency_table(
        self,
        methods: list[str],
        params:  list[str],
        flops: list[str],
        inference_time: list[str],
        accuracy_per_flop: list[str],
        caption: str = "Computational efficiency comparison",
        label:  str = "tab: efficiency",
    ) -> str:
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Params & FLOPs & Inference (ms) & Acc/FLOP \\",
            r"\midrule",
        ]

        for idx, method in enumerate(methods):
            line = (
                f"{method} & {params[idx]} & {flops[idx]} & "
                f"{inference_time[idx]} & {accuracy_per_flop[idx]} \\\\"
            )
            lines.append(line)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def save_table(self, table_content: str, filename: str) -> Path:
        filepath = self._output_dir / filename

        with open(filepath, "w") as f:
            f.write(table_content)

        return filepath