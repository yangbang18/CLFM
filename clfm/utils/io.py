import yaml


def load_yaml(args, key, yaml_path='configs/methods.yaml', yaml_data=None):
    if not key or key is None:
        return None

    assert yaml_path or yaml_data
    if yaml_data is None:
        yaml_data = yaml.full_load(open(yaml_path))
    
    assert key in yaml_data.keys(), f"`{key}` can not be found in {yaml_path}"

    specific_data = yaml_data[key]

    if 'inherit_from' in specific_data.keys():
        inherit_from = specific_data.pop('inherit_from')
        if isinstance(inherit_from, list):
            for new_key in inherit_from:
                load_yaml(args, key=new_key, yaml_path=yaml_path, yaml_data=yaml_data)    
        else:
            load_yaml(args, key=inherit_from, yaml_path=yaml_path, yaml_data=yaml_data)

    for k, v in specific_data.items():
        setattr(args, k, v)
        