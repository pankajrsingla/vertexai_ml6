## Deploying a Scikit-learn Pipeline + LightGBM Custom Model on Vertex AI for Explainable Online Predictions

[This notebook](https://github.com/pankajrsingla/vertexai_ml6/blob/main/xai_skl_lgbm.ipynb) and the [accompanying blogpost](https://medium.com/@pankaj_singla/vertex-ai-is-all-you-need-ba3a22c349dc) demonstrate how to use the Vertex AI SDK to train and deploy a custom model with Scikit-learn pipeline and LightGBM classifier for serving online predictions with explanations. 

Although Vertex AI has provided quite a few examples for deploying models built using Tensorflow/SKLearn/XGBoost, there are very few working examples explaining the deployment of custom container models, and almost none that show how to get explainable predictions for such models. This notebook is meant to bridge that gap.
