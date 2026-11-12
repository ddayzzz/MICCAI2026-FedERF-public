import torch
from torchmetrics.functional import f1_score, auroc
# from pytorch_lightning.metrics import Metric
from torchmetrics import Metric
from sklearn.metrics import (
    f1_score as sk_f1_score,
)
import torch.nn.functional as F
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
from sklearn.utils import resample

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if logits.size(-1) > 1:
            preds = logits.argmax(dim=-1)
        else:
            preds = (torch.sigmoid(logits) > 0.5).long()

        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class MultiLabelAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False, topk=1):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.topk = topk

    def update(self, logits, target):
        batch_size = logits.size(0)
        output, expected = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        assert (
                output.dim() <= 2
        ), "Output from model shouldn't have more than dim 2 for accuracy"
        assert (
                expected.dim() <= 2
        ), "Expected target shouldn't have more than dim 2 for accuracy"

        if output.dim() == 2:
            output = output.topk(self.topk, 1, True, True)[1].t().squeeze()

        # If more than 1
        # If last dim is 1, we directly have class indices
        if expected.dim() == 2 and expected.size(-1) != 1:
            expected = expected.topk(self.topk, 1, True, True)[1].t().squeeze()

        correct = (expected == output.squeeze()).sum().float()
        # return correct / batch_size

        self.correct += correct
        self.total += batch_size

    def compute(self):
        return self.correct / self.total


class AUROC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()

        if all_logits.size(-1) > 1:
            all_logits = torch.softmax(all_logits, dim=1)
            AUROC = auroc(all_logits, all_targets, num_classes=2)
        else:
            all_logits = torch.sigmoid(all_logits)
            AUROC = auroc(all_logits, all_targets)

        return AUROC


class F1_Score(Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.num_classes = num_classes

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
        if use_sigmoid:
            all_logits = torch.sigmoid(all_logits)
        F1_Micro = f1_score(all_logits, all_targets, average='micro')
        F1_Macro = f1_score(all_logits, all_targets, average='macro', num_classes=self.num_classes)
        F1_Samples = f1_score(all_logits, all_targets, average='samples')
        F1_Weighted = f1_score(all_logits, all_targets, average='weighted', num_classes=self.num_classes)
        return (F1_Micro, F1_Macro, F1_Samples, F1_Weighted)


def _convert_to_one_hot(expected, output):
    # This won't get called in case of multilabel, only multiclass or binary
    # as multilabel will anyways be multi hot vector
    if output.squeeze().dim() != expected.squeeze().dim() and expected.dim() == 1:
        expected = F.one_hot(
            expected.long(), num_classes=output.size(-1)
        ).float()
    return expected


class MultiLabelF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self):
        if type(self.logits) == list:
            scores = torch.cat(self.logits)
            expected = torch.cat(self.targets).long()
        else:
            raise NotImplementedError

        output = torch.sigmoid(scores)
        output = torch.round(output).cpu()
        expected = _convert_to_one_hot(expected, output).cpu()

        F1_Micro = sk_f1_score(expected, output, average="micro")
        F1_Macro = sk_f1_score(expected, output, average="macro")
        F1_Weighted = sk_f1_score(expected, output, average="weighted")
        F1_Samples = sk_f1_score(expected, output, average="samples")

        return (F1_Micro, F1_Macro, F1_Samples, F1_Weighted)


class check(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )

        self.logits.append(logits)
        self.targets.append(targets)

    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits).long()
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits.long()
            all_targets = self.targets.long()

        mislead = all_logits ^ all_targets
        accuracy = mislead.sum(dim=0)
        return accuracy


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total


class Scalar2(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar, num):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)

        self.scalar += scalar
        self.total += num

    def compute(self):
        return self.scalar / self.total

class C_Index(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("censorships", default=[], dist_reduce_fx="cat")
        self.add_state("event_times", default=[], dist_reduce_fx="cat")
        self.add_state("risks", default=[], dist_reduce_fx="cat")

    def update(self, logits, censorship, event_time):
        logits, censorships, event_times  = (
            logits.detach().to(self.device),
            censorship.detach().to(self.device),
            event_time.detach().to(self.device),
        )

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(S, dim=1).cpu().item()
        # self.logits.append(logits)
        self.censorships.append(censorships)
        self.risks.append(risk)
        self.event_times.append(event_times)

    def compute(self):
        if type(self.risks) == list:
            all_risk_scores = torch.tensor(self.risks)
            all_censorships = torch.cat(self.censorships)
            all_event_times = torch.cat(self.event_times)
        else:
            all_risk_scores = self.risks
            all_censorships = self.censorships
            all_event_times = self.event_times
        all_risk_scores = all_risk_scores.cpu().numpy()
        all_censorships = all_censorships.cpu().numpy()
        all_event_times = all_event_times.cpu().numpy()
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                             tied_tol=1e-08)[0]
        return c_index

class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total
