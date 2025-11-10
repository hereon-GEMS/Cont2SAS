# Contributing

Thank you for contributing to Cont2Sas!

Please follow the steps below to contribute to Sassena:

- Create a new [issue](https://codebase.helmholtz.cloud/DAPHNE4NFDI/sassena/-/issues) where you describe your intentions clearly and can gather feedback
- Get the source code: ``git clone https://codebase.helmholtz.cloud/DAPHNE4NFDI/continuum-to-scattering.git``
- Start from the development branch: ``git checkout develop``
- Make sure your ``develop`` branch is up to date with the repo's: ``git pull origin develop --rebase``
- Create a new feature branch from here with a descriptive name: ``git switch -c name`` where ``name`` could for example be ``fix/readme``
- code away!
   - ``git add`` the changed files you want to upload
   - ``git commit`` with a meaningful message
   - In your last commit of this series, include a [keyword](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests) to close your feature branch.
- Upload your feature branch: ``git push origin HEAD``
- Create a merge request of your feature branch into the ``develop`` branch

From time to time, we will merge the ``develop`` branch into the ``main`` branch.
