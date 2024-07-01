import hopsworks

def connect_to_hopsworks():
    project = hopsworks.login()

    fs = project.get_feature_store()

    return fs