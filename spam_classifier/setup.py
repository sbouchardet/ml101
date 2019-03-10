from setuptools import setup

setup(
    name='spam_classifier',
    version='0.1',
    py_modules=['main'],
    include_package_data=True,
    install_requires=[
        'Click',
        'pandas',
        'scikit-learn',
        'spacy'
        
    ],
    entry_points='''
        [console_scripts]
        tfidf=main:tfidf
        naive_bayes=main:naive_bayes
        split_dataset=main:split_dataset
    ''',
)