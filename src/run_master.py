def gen_dataset(model, dataset_root,  doe_file, index_range, other_params):
    doe = pd.read_csv(fr'{doe_file}')
    doe.index += 1

    if index_range is None:
        index_range = doe.index
        
    doe_lhs = np.loadtxt(doe_file, delimiter=',', skiprows=1)

    for id, row in doe.iterrows():
        if id in index_range:
            model( rootfile = f"{dataset_root}/{int(id)}/", 
                        **dict(row), **other_params)
                                     
if __name__ == "__main__":
    pass