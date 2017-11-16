#!/bin/bash

# Quick and Dirty Integration Test for the Jerb system
# Before running it, please erase the redis cache with 'flushdb'
# You may also want to erase any stored jerbs. 

WORKDIR="/home/ivar/test/"

cd $WORKDIR

# Test ID 
TESTID=`uuidgen`

###############################################################################
echo "Creating repo: hello"
jerb init hello

cd hello
echo "Hello, " > hello.txt
echo "Adding metadata..."
git add hello.txt
git notes add -f jerb_metadata -m "{\"user\": \"ivar\", \"parents\": [], \"description\": \"A simple jerb containing the file hello.txt.\", \"branch\": \"hello\", \"tags\": [\"example\", \"hello\", \"$TESTID\"]}"
git commit -m "Initial commit"

echo "Building hello.jerb..."
jerb jerb > ../hello.jerb


###############################################################################
cd $WORKDIR
echo "Creating repo: world"
jerb init world
cd world
echo "world!" > world.txt
git add world.txt
git notes add -f jerb_metadata -m "{\"user\": \"ivar\", \"parents\": [], \"description\": \"Another file, containing just world.txt\", \"branch\": \"world\", \"tags\": [\"example\", \"world\", \"$TESTID\"]}"
git commit -m "Initial commit" > /dev/null

echo "Building world.jerb..."
jerb jerb > ../world.jerb


###############################################################################

cd $WORKDIR
echo "Creating repo: helloworld"
jerb init helloworld
cd helloworld
jerb merge ../hello.jerb ../world.jerb


##############################################################################
# Testing searching
cd $WORKDIR

echo "Sharing hello.jerb and world.jerb..."
jerb share hello.jerb
jerb share world.jerb

echo "Searching for JID..."
jid=`jerb find "{\"branch\": \"hello\", \"tags\": \"$TESTID\"}"`

echo "Fetching and diffing JID: $jid"
diff hello.jerb <(jerb fetch $jid)


################################################################################
# Jerb Central
cd $WORKDIR

jerb init central
cd central
git fetch ../hello
git fetch ../world
git fetch ../helloworld



#https://stackoverflow.com/questions/10808345/how-to-add-additional-parents-to-old-git-commits
#git merge $intended_parent_1 $intended_parent_2
#git checkout $original_commit -- .
#git commit --amend -a -C $original_commit
