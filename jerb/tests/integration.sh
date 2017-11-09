#!/bin/bash

# Quick and Dirty Integration Test for the Jerb system
# Before running it, please erase the redis cache with 'flushdb'
# You may also want to erase any stored jerbs. 

WORKDIR=`/tmp/`

cd $WORKDIR

###############################################################################
echo "Creating repo: hello"
jerb init hello

cd hello
echo "Hello, " > hello.txt
echo "Adding metadata..."
git add hello.txt
git notes add -f jerb_metadata -m '{"user": "ivar", "parents": "", "description": "A simple jerb containing the file hello.txt.", "branch": "hello", "tags": ["example", "hello"]}'
git commit -m "Initial commit"

echo "Building hello.jerb..."
jerb jerb > ../hello.jerb
cd $WORKDIR

echo "Sharing hello.jerb..."
jerb share hello.jerb

echo "Searching for JID..."
jid=`jerb find '{"branch": "hello"}'`

echo "Fetching JID: $jid"
jerb fetch $jid > $jid.jerb

echo "Diffing"
diff hello.jerb $jid.jerb


###############################################################################

echo "Creating repo: world"
jerb init world
cd world
echo "world!" > world.txt
git add world.txt
git notes add -f jerb_metadata -m '{"user": "ivar", "parents": "", "description": "Another file, containing just world.txt", "branch": "world", "tags": ["example", "world"]}'
git commit -m "Initial commit"

echo "Building world.jerb..."
jerb jerb > ../world.jerb
cd $WORKDIR

echo "Sharing world.jerb..."
jerb share world.jerb

###############################################################################

echo "Creating repo: helloworld"
jerb init helloworld
cd helloworld
jerb find '{"tags": ["example"]}' | jerb merge
