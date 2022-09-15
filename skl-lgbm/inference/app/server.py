import os
import google.cloud.storage as gs
import pickle
import pandas as pd

class LGBM:
    def __init__(self):
        MODEL_FILENAME = "skl_lgbm.pkl"
        
        # Downloading the saved model file:
        aip_storage_uri = os.environ['AIP_STORAGE_URI']  # gs-path to directory with model artifacts
        # This works only for docker. If you want to run the file locally, specify the exact bucket name like below:
        # aip_storage_uri = 'gs://my_bucket'  # gs-path to directory with model artifacts
        # Below are some additional steps required only for a custom container
        # using additional modules such as LightGBM.
        aip_storage_uri = aip_storage_uri.replace('gs://', '')
        aip_storage_uri = aip_storage_uri.rstrip('/')
        first_slash = aip_storage_uri.find('/')
        if first_slash > 0:
            bucket_name = aip_storage_uri[:first_slash] 
            model_path = aip_storage_uri[first_slash+1:] + '/' + MODEL_FILENAME
        else:
            bucket_name = aip_storage_uri
            model_path = MODEL_FILENAME 
        storage_client = gs.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.get_blob(model_path)
        blob.download_to_filename(MODEL_FILENAME)
        
        self.model = pickle.load(open(MODEL_FILENAME, 'rb'))
        
    def predict(self, features:dict):
        df = pd.DataFrame(features, index=[0])

        # Prediction from saved model
        prediction = self.model.predict(df).tolist()
        return {'Survived': prediction}
