
#### 1.  **What should I do if the cann-cla/no label is added to my PR?**

This label indicates that among the commits in this PR, some contributors have not signed the CANN community Contributor License Agreement (CLA). The signing link can be found in the PR comment area.

(1) If you are contributing as an individual, select "Individual CLA Signing".

(2) If you are contributing on behalf of a corporate, select "Corporate CLA Signing".

(3) If you are contributing as an employee of a corporate that has already signed the corporate CLA, select "Corporate Contributor Registration".

After signing, you will receive an email with the subject "Signing CLA on project of xx". Contact the Corporation Managers mentioned in the email for approval. Once approved, comment `/check-cla` in the PR comment area to re-trigger the CLA check, and the label will change to `cann-cla/yes`. The CLA check uses the committer email in the commit information as the verification credential. This email can be queried using `git log --pretty=fuller`.

<table>
<tbody><tr>
<th>Scenario</th>
<th>Preference</th>
<th>Solution</th>
</tr>
<tr>
<td>Commit email identical to the GitCode submission email</td>
<td>This identical email address</td>
<td>Use this email address to sign the CLA.</td>
</tr>
<tr>
<td rowspan="2">Commit email different from the GitCode submission email</td>
<td>Commit email</td>
<td>Change the Gitcode submission email address to the commit email address. On the GitCode personal settings page, add the commit email address and set it as the submission email address. Then, sign the CLA.</td>
</tr>
<tr>
<td>GitCode submission email</td>
<td>On the local host where Git is running, run the `git config --global user.name **` and `git config --global user.email **` commands to change the commit email address of Git to the GitCode submission email address. Then, sign the CLA.</td>
</tr>
</tbody>
</table>

#### 2.  Why can't I fork the CANN/abc repository to my account?

This issue typically occurs because a repository with the same name `abc` already exists under your personal account. For example, you may have previously forked a repository named `abc` from the CANN organization. Since GitCode uses your personal account name plus the repository name for addressing, duplicate repository names under your personal account are not allowed.

Solution: Rename or change the path of the existing repository under your personal account, then fork the `CANN/abc` repository again.

#### 3.  What are the differences between protected and unprotected branches?

A protected branch allow specific roles or members to have push and merge permissions for this branch. Unprotected branches do not support this.

#### 4.  Can CANN developers in the community directly push code to repositories?

CANN developers are not allowed to directly push code to the community repository. Only the repository administrators are allowed to do so.
If a CANN developer wants to contribute code, they must fork the community repository to their personal account and contribute code by submitting pull requests.

#### 5.  **What is the difference between directly pushing code to a repository and merging code via `/lgtm` or `/approve` comments?**

Using Git commands to directly push code to a repository bypasses necessary reviews, introducing risks. For example, when a file to upload is too large for a personal repository, you need push it directly to an unprotected branch in the repository, then merge the changes into a protected branch.

Merging code via `/lgtm` or `/approve` comments adds a review step to the process. This ensures that at least one committer other than the submitter has approved the code before it is merged. Even if the submitter is themselves a committer, another committer's approval is still required.

#### 6.  **What commands can I use in the comments of CANN community repositories and what are their functions?**

For details about the supported commands, see [CANN Community Comment Commands](infra-command.md).

#### 7.  **Why is CI build not triggered after I submit a PR?**

CI build will not be triggered in the following scenarios:

- Scenario 1: Due to network issues or system task scheduling issues, the webhook notification event sent from the code repository may not have reached the target service in time. In this case, you can re-trigger it by commenting `/compile` in the PR comment area.

- Scenario 2: The PR was submitted shortly after the code repository was created. At this point, the CI build project has not yet been created on the Jenkins server, so the CI build cannot be triggered, and commenting `/compile` will not work. In this case, please wait a moment for the system to automatically build the project.
