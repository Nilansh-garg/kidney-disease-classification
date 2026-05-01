import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    

__version = "0.0.1"

REPO_NAME = "Kidney_Disease_classification_MLFlow-DVC"
AUTHOR_USER_NAME = "Nilansh Garg"
SRC_REPO = "cnn_classifier"
AUTHOR_EMAIL = "nilanshgarg13@gmail.com"


setuptools.setup(
    name = SRC_REPO,
    version = __version,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "A machine learning project for classifying kidney diseases using CNN, MLFlow, and DVC.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = f"http://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
        },
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src")
)