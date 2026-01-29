<!-- omit in toc -->
# Contributing to Cont2SAS

Copyright (C) 2025  Helmholtz-Zentrum-Hereon

Authors: Arnab Majumdar and Sebastian Busch

<!-- omit in toc -->
## Introduction

The project team of [Cont2SAS](https://github.com/hereon-GEMS/Cont2SAS) thanks you for taking the time to contribute!

All types of contributions are encouraged and valued. See the [table of contents](#table-of-contents) for different ways to help and details about how the project team handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions.

> If you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project.
> - Refer this project in your project's readme.
> - Mention the project at local meetups and tell your friends/colleagues.

<!-- omit in toc -->
## Table of contents

- [Code of conduct](#code-of-conduct)
- [I have a question](#i-have-a-question)
- [I want to contribute](#i-want-to-contribute)
  - [Before contribution](#before-contribution)
  - [Your first code contribution](#your-first-code-contribution)
- [Styleguides](#styleguides)


## Code of conduct

This project and everyone participating in it is governed by the
[Cont2SAS code of conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.


## I have a question

> If you want to ask a question, we assume that you have read the available [documentation](./docu/index.md).

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [issue](https://github.com/hereon-GEMS/Cont2SAS/issues/new).
- Provide as much context as you can.

## I want to contribute

<!-- omit in toc  -->
> ### Legal notice 
> When contributing to this project, you must agree that you have authored 100\% of the content, that you have the necessary rights to the content and that the content you contribute may be provided under the project licence.

### Before contribution

<!-- omit in toc  -->
#### Bug report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report.

<!-- omit in toc  -->
#### Enhancement suggestion

Find out whether the suggested enhancement fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of Cont2SAS users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

<!-- omit in toc  -->
#### General checklist

- Make sure that you are using the latest version.
- Make sure that you have read the [documentation](./docu/index.md).
- Check [existing issues](https://github.com/hereon-GEMS/Cont2SAS/issues).

### Your first code contribution

Please follow the steps below to contribute to Cont2SAS:

- Create a new [issue](https://github.com/hereon-GEMS/Cont2SAS/issues/new), where you describe your intentions clearly and can gather feedback.
- Create a fork of the project in GitHub website.
- Clone the forked project using ``git clone``. The forked project link should look like ``git@github.com:majumdararn/Cont2SAS.git``, where ``majumdararn`` should be replaced by your user name in GitHub.
- Inside cloned project, add [original repository](https://github.com/hereon-GEMS/Cont2SAS) as upstream using ``git remote add upstream git@github.com:hereon-GEMS/Cont2SAS.git``.
- Fetch from the original repository using ``git fetch upstream``.
- Merge all changes from the develop branch of original repository to the clone of forked project using ``git merge upstream/develop``.
- Create a new feature branch from main with a descriptive name using ``git switch -c name``, where ``name`` could for example be ``fix/readme``.
- Code away!
   - Use ``pytest -v`` to check if all tests pass (add new test routine if a new simulation model is added).
   - Use ``git add`` to upload the changed files.
   - Use ``git commit`` to commit the uploaded change with a meaningful message.
- Upload your feature branch using ``git push origin HEAD``. 
- Create a pull request to merge the changes from the feature branch of forked project to develop branch of original project in GitHub.
- Mention ``see #$ISSUE_NUM`` to link with issue or ``fix #$ISSUE_NUM`` to close the branch, where ``$ISSUE_NUM`` should be repled by the relevant issue number.
- Once the pull request is approved and changes are impleted in the development branch of original repository, delete the feature branch in forked project using following steps:
  - Go the develop branch using ``git checkout develop``.
  - Delete the local feature branch using ``git branch -d name``, where ``name`` should be replaced by the name of feature-branch.
  - Delete the remote feature branch using ``git push origin --delete name``, where ``name`` should be replaced by the name of feature-branch.

From time to time, the project team will merge the ``develop`` branch into the ``main`` branch.

## Styleguides

<!-- omit in toc  -->
### Python code  
Please ensure your code follows the [pylint style guide](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html)
 and passes all linting checks before submitting a merge request.

<!-- omit in toc  -->
### Commit messages 
Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

<!-- omit in toc  -->
## Attribution
This guide was generated by [contributing.md](https://contributing.md/generator). The authors performed necessary edits and take full responsibility for the the content.