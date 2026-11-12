

def split_data(csv_file, n_fold):
    data = pd.read_csv(csv_file, index_col=0)