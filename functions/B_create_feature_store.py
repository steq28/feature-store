import hopsworks

def connect_to_hopsworks():
    project = hopsworks.login()

    fs = project.get_feature_store()

    return project, fs

def create_feature_group(name, version, description, primary_key, online_enabled, fs, df, feature_descriptions):
    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        description=description,
        primary_key=primary_key,
        online_enabled=online_enabled,
    )

    # Insert data into feature group
    fg.insert(df)

    for desc in feature_descriptions: 
        fg.update_feature_description(desc["name"], desc["description"])

def close_connection(fs):
    fs.close()