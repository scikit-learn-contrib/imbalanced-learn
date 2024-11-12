from dataclasses import dataclass, field

import sklearn
from sklearn.utils.fixes import parse_version
from .fixes import _dataclass_args

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)

if sklearn_version >= parse_version("1.6"):
    from sklearn.utils._tags import (
        TargetTags,
        TransformerTags,
        ClassifierTags,
        RegressorTags,
        InputTags,
    )

    @dataclass(**_dataclass_args())
    class InputTags(InputTags):
        dataframe: bool = True

    @dataclass(**_dataclass_args())
    class SamplerTags:
        """Tags for the sampler.

        Parameters
        ----------
        sample_indices : bool, default=False
            Whether the sampler returns the indices of the samples that were
            selected.
        """

        sample_indices: bool = False


@dataclass(**_dataclass_args())
class Tags:
    """Tags for the estimator.

    See :ref:`estimator_tags` for more information.

    Parameters
    ----------
    estimator_type : str or None
        The type of the estimator. Can be one of:
        - "classifier"
        - "regressor"
        - "transformer"
        - "clusterer"
        - "outlier_detector"
        - "density_estimator"

    target_tags : :class:`TargetTags`
        The target(y) tags.

    transformer_tags : :class:`TransformerTags` or None
        The transformer tags.

    classifier_tags : :class:`ClassifierTags` or None
        The classifier tags.

    regressor_tags : :class:`RegressorTags` or None
        The regressor tags.

    sampler_tags : :class:`SamplerTags` or None
        The sampler tags.

    array_api_support : bool, default=False
        Whether the estimator supports Array API compatible inputs.

    no_validation : bool, default=False
        Whether the estimator skips input-validation. This is only meant for
        stateless and dummy transformers!

    non_deterministic : bool, default=False
        Whether the estimator is not deterministic given a fixed ``random_state``.

    requires_fit : bool, default=True
        Whether the estimator requires to be fitted before calling one of
        `transform`, `predict`, `predict_proba`, or `decision_function`.

    _skip_test : bool, default=False
        Whether to skip common tests entirely. Don't use this unless
        you have a *very good* reason.

    input_tags : :class:`InputTags`
        The input data(X) tags.
    """

    estimator_type: str | None
    target_tags: TargetTags
    transformer_tags: TransformerTags | None
    classifier_tags: ClassifierTags | None
    regressor_tags: RegressorTags | None
    sampler_tags: SamplerTags | None
    array_api_support: bool = False
    no_validation: bool = False
    non_deterministic: bool = False
    requires_fit: bool = True
    _skip_test: bool = False
    input_tags: InputTags = field(default_factory=InputTags)
