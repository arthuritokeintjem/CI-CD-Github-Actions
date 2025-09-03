from sklearn.datasets import load_iris

def test_dataset_shape():
    data = load_iris()
    assert data.data.shape == (150, 4)
    assert len(data.target) == 150