#!/bin/sh
#
# This script scans subdirectories for files <modulename>.module
# and creates the file modules.list and dolfin_modules.h

MODULES="modules.list"
HEADER="../kernel/main/dolfin_modules.h"

rm -f $MODULES
rm -f $HEADER

# Prepare file modules.list
echo "# This file is automatically generated by the script scanmodules.sh" > $MODULES

# Find all modules
DIRS=`ls`
COUNT="0"
for d in $DIRS; do
	 if [ -r $d/$d.module ]; then
		  echo $d >> $MODULES
        COUNT=`echo $COUNT | awk '{ print $1+1 }'`
    fi
done
echo "Found $COUNT modules in src/modules:"

# Generate code
echo "// This code is automatically generated by the script scanmodules.sh" >> $HEADER
echo "// Modules are automatically detected from src/modules/ as directories" >> $HEADER
echo "// containing a file src/modules/<name>/<name>.module." >> $HEADER
echo " " >> $HEADER
for d in `cat $MODULES | grep -v '#'`; do
    # Find NAME from config file
	 NAME=`cat $d/$d.module | grep NAME | cut -d'"' -f2`
	 if [ 'x'$NAME = 'x' ]; then
		  echo "Unable to find NAME for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi
	 if [ $NAME != $d ]; then
		  echo "Module name $NAME does not match name of directory $d"
		  exit 1
    fi
    # Find KEYWORD from config file (can contain more than one word!)
	 KEYWORD=`cat $d/$d.module | grep KEYWORD | cut -d'"' -f2`
	 CHECK=`echo $KEYWORD | awk '{ print $1 }'`
	 if [ 'x'$CHECK = 'x' ]; then
		  echo "Unable to find KEYWORD for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi
	 # Find SOLVER from config file
 	 SOLVER=`cat $d/$d.module | grep SOLVER | cut -d'"' -f2`
	 if [ 'x'$SOLVER = 'x' ]; then
		  echo "Unable to find SOLVER for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi
	 # Find SETTINGS from config file
 	 SETTINGS=`cat $d/$d.module | grep SETTINGS | cut -d'"' -f2`
	 if [ 'x'$SETTINGS = 'x' ]; then
		  echo "Unable to find SETTINGS for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi

	 # Write a nice message
	 echo "  $NAME ($KEYWORD)"
	 # Include files
	 echo "#include \"$SOLVER.hh\"" >> $HEADER
	 echo "#include \"$SETTINGS.hh\"" >> $HEADER
done
echo "" >> $HEADER
echo "#include <string.h>" >> $HEADER
echo "" >> $HEADER
echo "#define DOLFIN_MODULE_COUNT $COUNT" >> $HEADER
echo "" >> $HEADER
echo "Solver * dolfin_module_problem(const char *keyword, Grid *grid)" >> $HEADER
echo "{" >> $HEADER
for d in `cat $MODULES | grep -v '#'`; do
    # Find KEYWORD from config file
	 KEYWORD=`cat $d/$d.module | grep KEYWORD | cut -d'"' -f2`
	 CHECK=`echo $KEYWORD | awk '{ print $1 }'`
	 if [ 'x'$CHECK = 'x' ]; then
		  echo "Unable to find KEYWORD for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi
	 # Find SOLVER from config file
 	 SOLVER=`cat $d/$d.module | grep SOLVER | cut -d'"' -f2`
	 if [ 'x'$SOLVER = 'x' ]; then
		  echo "Unable to find SOLVER for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi
	 echo "    if ( strcasecmp(keyword,\"$KEYWORD\") == 0 )" >> $HEADER
	 echo "        return new $SOLVER(grid);" >> $HEADER
done
echo "" >> $HEADER
echo "    display->Error(\"Could not find any matching solver for problem \\\"%s\\\".\",keyword);" >> $HEADER
echo "" >> $HEADER
echo "    return 0;" >> $HEADER
echo "}" >> $HEADER
echo "" >> $HEADER
echo "Settings * dolfin_module_settings(const char *keyword)" >> $HEADER
echo "{" >> $HEADER
for d in `cat $MODULES | grep -v '#'`; do
    # Find KEYWORD from config file
	 KEYWORD=`cat $d/$d.module | grep KEYWORD | cut -d'"' -f2`
    CHECK=`echo $KEYWORD | awk '{ print $1 }'`
	 if [ 'x'$CHECK = 'x' ]; then
		  echo "Unable to find KEYWORD for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi
	 # Find SETTINGS from config file
 	 SETTINGS=`cat $d/$d.module | grep SETTINGS | cut -d'"' -f2`
	 if [ 'x'$SETTINGS = 'x' ]; then
		  echo "Unable to find SETTINGS for module $d in file src/modules/$d/$d.module"
		  exit 1
    fi
	 echo "    if ( strcasecmp(keyword,\"$KEYWORD\") == 0 )" >> $HEADER
	 echo "        return new $SETTINGS();" >> $HEADER
done
echo "" >> $HEADER
echo "    display->Error(\"Could not find any matching solver for problem \\\"%s\\\".\",keyword);" >> $HEADER
echo "" >> $HEADER
echo "    return 0;" >> $HEADER
echo "}" >> $HEADER
