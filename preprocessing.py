import pandas as pd

# def get_train_file_path(image_id):
#     return f'/raid/yiw/seti/input/train/{image_id[0]}/{image_id}.npy'

# df = pd.read_csv('input/train_labels.csv')
# df.drop(columns=['file_path'])
# df['file_path'] = df['id'].apply(get_train_file_path)

# df.to_csv('input/train_labels.csv', index=False)

def get_test_file_path(image_id):
    return f'/raid/yiw/seti/input/test/{image_id[0]}/{image_id}.npy'

df = pd.read_csv('input/submission.csv')
df.drop(columns=['file_path'])
df['file_path'] = df['id'].apply(get_test_file_path)

df.to_csv('input/submission.csv', index=False)