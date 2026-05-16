from autocluster_api import AutoClusterAPI

api = AutoClusterAPI(data="pipeline/raw-data.csv")
api.serve(port=8181)