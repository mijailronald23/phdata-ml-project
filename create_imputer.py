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


def main():
    """Fit and export the offline KNN imputation artifact."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, _y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42
    )

    artifact = fit_knn_imputer_artifact(
        reference_frame=x_train,
        feature_names=list(x_train.columns),
        property_features=REQUEST_PROPERTY_FEATURES,
    )

    output_dir = ArtifactPaths().features_path.parent
    pathlib.Path(output_dir).mkdir(exist_ok=True)
    output_path = pathlib.Path(output_dir) / "knn_imputer.pkl"
    with output_path.open("wb") as artifact_file:
        pickle.dump(artifact, artifact_file)

    print(f"Saved KNN imputer artifact to {output_path}")


if __name__ == "__main__":
    main()
