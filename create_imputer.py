import pathlib
import pickle

from sklearn import model_selection

from create_model import DEMOGRAPHICS_PATH
from create_model import SALES_COLUMN_SELECTION
from create_model import SALES_PATH
from create_model import load_data
from preprocessing.imputation import fit_knn_imputer_artifact
from services.inference import REQUEST_PROPERTY_FEATURES
from utils.loader import ArtifactPaths


def main() -> None:
    """Fit and export the offline KNN imputation artifact.

    The artifact is fit on the training reference split only so the target and
    evaluation records are excluded from preprocessing estimation.
    """
    feature_frame, target_values = load_data(
        SALES_PATH,
        DEMOGRAPHICS_PATH,
        SALES_COLUMN_SELECTION,
    )
    training_features, _test_features, _training_targets, _test_targets = (
        model_selection.train_test_split(
            feature_frame,
            target_values,
            random_state=42,
        )
    )

    # ``price`` is never part of the imputer fit; only training-reference model
    # features are used so the artifact cannot leak target information.
    imputation_artifact = fit_knn_imputer_artifact(
        reference_frame=training_features,
        feature_names=list(training_features.columns),
        property_features=REQUEST_PROPERTY_FEATURES,
    )

    artifact_directory = ArtifactPaths().features_path.parent
    pathlib.Path(artifact_directory).mkdir(exist_ok=True)
    output_path = pathlib.Path(artifact_directory) / "knn_imputer.pkl"
    with output_path.open("wb") as artifact_file:
        pickle.dump(imputation_artifact, artifact_file)

    print(f"Saved KNN imputer artifact to {output_path}")


if __name__ == "__main__":
    main()
