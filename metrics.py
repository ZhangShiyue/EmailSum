import logging
import numpy as np
from rouge import rouge_scorer
from rouge import scoring
from bert_score import score
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def rouge(targets, predictions, score_keys=None, use_stemmer=True):
    """Computes rouge score.
    Args:
      targets: list of strings
      predictions: list of strings
      score_keys: list of strings with the keys to compute.
    Returns:
      dict with score_key: myrouge score across all targets and predictions
    """

    if score_keys is None:
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(score_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for prediction, target in zip(predictions, targets):
        target = target
        prediction = prediction
        aggregator.add_scores(scorer.score(target=target, prediction=prediction))
    result = aggregator.aggregate()
    res_str = '\n'.join(["%s = %.2f, 95%% confidence [%.2f, %.2f]" %
                        (key, result[key].mid.fmeasure * 100, result[key].low.fmeasure * 100,
                         result[key].high.fmeasure * 100,) for key in score_keys])

    logger.info(res_str)
    return {key: result[key].mid.fmeasure * 100 for key in score_keys}, res_str


def extScore(sources, predictions, use_stemmer=True):
    """Computes rouge score.
        Args:
          sources: list of strings
          predictions: list of strings
        Returns:
          dict with score_key: rouge score across all targets and predictions
        """
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=use_stemmer)
    precisions = []
    for prediction, source in zip(predictions, sources):
        source = source
        prediction = prediction
        res = scorer.score(target=source, prediction=prediction)
        precisions.append(res["rougeLsum"].precision)
    precision = 1. - np.mean(precisions)
    res = {"ext_rouge2_prec": precision*100}
    res_str = '\n'.join(["%s = %.2f" %
                         (key, res[key]) for key in res])

    logger.info(res_str)
    return res, res_str


def bertScore(refs, cands, rescale_with_baseline=True):
    P, R, F1 = score(cands, refs, lang="en", rescale_with_baseline=rescale_with_baseline)
    res = {"bertScore": F1.mean().item()*100}
    res_str = '\n'.join(["%s = %.2f" %
                         (key, res[key]) for key in res])
    logger.info(res_str)
    return res, res_str
