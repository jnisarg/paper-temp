import torch
from tabulate import tabulate
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Metrics:
    def __init__(
        self,
        num_classes,
        class_names=None,
        table_fmt="fancy_grid",
        missing_val="-",
        ignore_index=255,
        eps=1e-6,
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.table_fmt = table_fmt
        self.missing_val = missing_val
        self.ignore_index = ignore_index
        self.eps = eps
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        ).cuda()
        self.mAP= MeanAveragePrecision(iou_type="bbox").cuda()
        self.metrics = {}

    def update(self, pred, target):
        #for OD metrics
        target_od= {}
        target_od['boxes']= target[1][0]
        target_od['labels']= target[2][0]
        target_od= [target_od]
        pred_od= [{
            'boxes': torch.tensor([[258.0, 41.0, 606.0, 285.0]]).cuda(),
            'scores': torch.tensor([0.536]).cuda(),
            'labels': torch.tensor([0]).cuda()
        }] #replace this with actual predictions
        self.mAP.update(pred_od, target_od)


        #for SD Metrics
        pred_sd, target_sd = self.prepare_input(pred[0], target[4])
        self.confusion_matrix += torch.bincount(
            self.num_classes * target_sd + pred_sd, minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes)

    def prepare_input(self, pred, target):
        if pred.dim() > 3:
            pred = torch.argmax(pred, dim=1).view(-1)
        else:
            pred = pred.view(-1)
        target = target.view(-1)
        mask = target != self.ignore_index
        return pred[mask], target[mask]

    def collect(self):
        TP = self.confusion_matrix.diag()
        FP = self.confusion_matrix.sum(0) - TP
        FN = self.confusion_matrix.sum(1) - TP

        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)
        f1_score = 2 * (precision * recall) / (precision + recall + self.eps)
        iou = TP / (TP + FP + FN + self.eps)

        class_weight = self.confusion_matrix.sum(1) / self.confusion_matrix.sum()

        self.metrics.update(
            {
                "class_weight": class_weight,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "mAP": self.mAP.compute()['map'].item(),
                "mAP_50": self.mAP.compute()['map_50'].item(),
                "mAP_75": self.mAP.compute()['map_75'].item()

            }
        )

    def create_summary_table(self, metrics):
        class_names = self.class_names if self.class_names else range(self.num_classes)

        tabular_data = []

        for idx, name in enumerate(class_names):
            tabular_data.append(
                [
                    name,
                    *[
                        (
                            f"{metrics[field][idx] * 100:.2f}%"
                            if field != "class_weight"
                            else f"{metrics[field][idx]:.6f}"
                        )
                        for field in list(metrics.keys())[:-3]
                    ],
                ]
            )

        tabular_data.append(
            [
                "Mean",
                *[
                    (
                        f"{metrics[field].mean() * 100:.2f}%"
                        if field != "class_weight"
                        else None
                    )
                    for field in list(metrics.keys())[:-3]
                ],
            ]
        )

        headers = ["CLASS NAMES" if self.class_names else "CLASS ID"] + [
            field.upper() for field in metrics
        ]

        return tabulate(
            tabular_data,
            headers,
            tablefmt=self.table_fmt,
            missingval=self.missing_val,
        )

    def __str__(self):
        self.collect()
        return self.create_summary_table(self.metrics)
