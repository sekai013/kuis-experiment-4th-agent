from setuptools import setup, find_packages

setup(
        name='kuis-ex4-agent',
        version='1.0',
        packages=find_packages(),
        install_requires = ['click', 'numpy', 'cvxopt', 'matplotlib'],
        entry_points = """
            [console_scripts]
            svm=scripts.svm:cli
            auction=scripts.auction_client:cli
        """,
        zip_safe = False,
        classifiers=[
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Operating System :: MacOS :: MacOS X',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7'
            ])
