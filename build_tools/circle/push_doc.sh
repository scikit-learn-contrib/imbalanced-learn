#!/bin/bash
# This script is meant to be called in the "deploy" step defined in
# circle.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the circle.yml in the top level folder of the project.

if [ "$CIRCLE_BRANCH" = "master" ]
then
    dir=dev
else
    # Strip off .X
    dir="${CIRCLE_BRANCH::-2}"
fi

MSG="Pushing the docs to $dir/ for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

cd $HOME
if [ ! -d $DOC_REPO ];
then git clone --depth 1 --no-checkout "git@github.com:"$ORGANIZATION"/"$DOC_REPO".git";
fi
cd $DOC_REPO
git config core.sparseCheckout true
echo $dir > .git/info/sparse-checkout
git checkout gh-pages
git reset --hard origin/gh-pages
git rm -rf $dir/ && rm -rf $dir/
cp -R $HOME/imbalanced-learn/doc/_build/html $dir
touch $dir/.nojekyll
git config --global user.email $EMAIL
git config --global user.name $USERNAME
git config --global push.default matching
git add -f $dir/
git commit -m "$MSG" $dir
git push

echo $MSG
