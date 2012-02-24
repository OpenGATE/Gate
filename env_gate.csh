########### Comments
#
# This script performs two tasks:
# 1. it launches the Geant4 configuration script
# 2. it defines the environment variables required by GATE
#
# The script assumes two prerequisites:
# 1. that the user has already defined the variable G4INSTALL
# 2. that there is a configuration script 'env.sh' in the directory $G4INSTALL
# 3. the variable G4VERSION needs to be define: 9.1 or 9.2
# 
# You will need to make sure that both conditions are met before you can use this script 
#
###########

setenv G4ANALYSIS_USE_GENERAL 1
setenv G4_USE_G4BESTUNIT_FOR_VERBOSE 1

########### Geant4 configuration stage
#
# In the section below, we make sure that we have access to a script
# 'env.sh' in the directory $G4INSTALL, then launch this script
#
# If you have configured Geant4 by some other mean, you can actually 
# skip this stage and go directly to the Gate configuration stage 
#
##########

echo ""

# We test that the variables G4INSTALL ans G4VERSION are set 
if ( ! ${?G4INSTALL} ) then
  echo "The environment variable G4INSTALL is not defined yet."
  echo "You must define G4INSTALL before you can configure (and compile) GATE."
  echo "Aborting"
  exit
else if ( ! ${?G4VERSION} ) then
  echo "The environment variable G4VERSION is not defined yet."
  echo "You must define G4VERSION before you can configure (and compile) GATE."
  echo "Aborting"
  exit
endif
  
echo "The current value of G4INSTALL is: ${G4INSTALL}"
echo "The current value of G4VERSION is: ${G4VERSION}"
echo ""
echo "Launching Geant4 configuration script "$G4INSTALL"/env.csh"
echo ""

# We test that there truly is a directory $G4INSTALL
if ( ! -d $G4INSTALL )  then
  echo "This does not seem to be a valid directory."
  echo "You must define G4INSTALL before you can configure (and compile) GATE."
  echo "Aborting"
  exit
endif

# We test that there is a Geant4 configuration script 'env.csh' in the directory $G4INSTALL
if ( ! -f $G4INSTALL"/env.csh" )  then
  echo "I could not find a valid Geant4 script file env.sh in "$G4INSTALL
  echo "You must provide a path with a valid Geant4 configuration script env.csh."
  echo "Aborting"
  exit
endif

# Launch the Geant4 configuration script
source $G4INSTALL"/env.csh"

########### End of Geant4 configuration stage


########### Gate configuration stage
#
# In the section below, we configure the GATE environment variables
# Those are essentially related to the various data-output options
#
# Most options can probably be left as they are
# However, you may need to check the variables G4ANALYSIS_USE_ROOT and
# G4ANALYSIS_USE_ROOT_PLOTTER, since both variables enable options that rely 
# on the presence of the ROOT libraries. If these libraries are not installed on 
# your system, you need to disable these options (i.e. comment the lines out)
#
##########

echo ""
echo "Configuring GATE options"
echo ""

# Set the working directories
setenv G4WORKDIR '.'
echo "G4WORKDIR set to $G4WORKDIR"
setenv G4TMP $G4WORKDIR"/tmp"
echo "G4TMP set to $G4TMP"
setenv G4BIN $G4WORKDIR"/bin"
echo "G4BIN set to $G4BIN"

# Enable the use of analysis and data-output 
# This line should normally always be left as it is
setenv G4ANALYSIS_USE_GENERAL 1
if ( ${?G4ANALYSIS_USE_GENERAL} )  then
  echo "Data analysis and output features enabled"
else
  echo "Data analysis and output features disabled"
endif


# Enable the use of the ASCII output file
# Comment this line if you want to disable this output (to speed-up the simulations for instance)
setenv G4ANALYSIS_USE_FILE 1
if ( ${?G4ANALYSIS_USE_FILE} )  then
  echo "Ascii data output enabled"
else
  echo "Ascii data output disabled"
endif

# Enable the use of optical transport in GATE
# We check if xml2-config is installed and present in the path: this is needed for compilation of
# GATE with optical transport.
# Comment this line if you want to disable this, for example when it is not needed and when there
# are problems with libXML2. 
unsetenv GATE_USE_OPTICAL
#setenv GATE_USE_OPTICAL 1
if ( ${?GATE_USE_OPTICAL} ) then
  
#  if [ -X xml2-config ] ; then
  if ( ! -f $PATH"/xml2-config" ) then
    echo "Transport and generation of optical photons is enabled"
  else
    echo "WARNING: xml2-config was not found in the path."
    echo "         transport and generation of optical photons has been disabled"
    echo " "
    echo "         When optical transport is needed: install the libXML2 development packages "
    echo "         (which should include xml2-config) and rerun the configuration script. It is "
    echo "         also possible to set the linker and compiler flags for libXML2 by hand in "
    echo "         GNUmakefile and set GATE_USE_OPTICAL to 1 after running this configuration "
    echo "         script."
    setenv GATE_USE_OPTICAL 0
  endif

else
  echo "Transport and generation of optical photons is disabled"
endif

# We check whether the ROOT home-directory variable has been set
# If that's the case, we assume that ROOT is installed on this system and enable 
# the ROOT-based data output and plotter
if ( ${?ROOTSYS} ) then
  # Enable the use of ROOT-based features (data output and optionally plotting)
  # Comment this line if you don't want to use ROOT 
  setenv G4ANALYSIS_USE_ROOT 1
  # Enable the use of the ROOT-based real-time plotter
  # Comment this line if you don't want to use the ROOT plotter
  # export G4ANALYSIS_USE_ROOT_PLOTTER=1
  unsetenv G4ANALYSIS_USE_ROOT_PLOTTER
  # Add the path for the ROOT shared libraries if not yet done  
  if ( -d $ROOTSYS"/lib" ) then
    if ( ${?LD_LIBRARY_PATH} ) then
      echo "Checking your LD_LIBRARY_PATH variable..."
      echo $LD_LIBRARY_PATH | grep ${ROOTSYS}/lib > /dev/null
      if ( ! $status )  then
        echo " ==> OK: ${ROOTSYS}/lib added to your LD_LIBRARY_PATH"  
        setenv LD_LIBRARY_PATH $ROOTSYS/lib:$LD_LIBRARY_PATH
      else
        echo " ==> OK: ${ROOTSYS}/lib is already in your LD_LIBRARY_PATH"
      endif
    else
      echo " ==> ${ROOTSYS}/lib added to your LD_LIBRARY_PATH"  
      setenv LD_LIBRARY_PATH ${ROOTSYS}/lib
    endif
  else
    echo "WARNING: there is no ${ROOTSYS}/lib directory"
    echo "         you should check that ROOT shared libraries are included in LD_LIBRARY_PATH"
  endif
endif
if ( ${?G4ANALYSIS_USE_ROOT} ) then
  echo "Use of ROOT enabled"
else
  echo "Use of ROOT disabled"
endif
if ( ${?G4ANALYSIS_USE_ROOT_PLOTTER} ) then
  echo "ROOT real-time plotter enabled"
else
  echo "ROOT real-time plotter disabled"
endif

# We check whether the LMF home-directory variable has been set
# If that's the case, we assume that the LMF library is installed on this system and enable 
# the LMF data output
if ( ${?LMF_HOME} ) then
  # Enable the use of the LMF output file
  # Comment this line if you want to disable this output (to speed-up the simulations for instance)
  setenv GATE_USE_LMF 1 
  if ( -d ${LMF_HOME}"/lib" ) then
    if ( ${?LD_LIBRARY_PATH} ) then
      echo "Checking your LD_LIBRARY_PATH variable..."
      echo $LD_LIBRARY_PATH | grep ${LMF_HOME}/lib > /dev/null
      if ( $status ) then
        echo " ==> OK: ${LMF_HOME}/lib added to your LD_LIBRARY_PATH"  
        setenv LD_LIBRARY_PATH ${LMF_HOME}/lib:$LD_LIBRARY_PATH
      else
        echo " ==> OK: ${LMF_HOME}/lib is already in your LD_LIBRARY_PATH"
      endif
    else
      echo " ==> ${LMF_HOME}/lib added to your LD_LIBRARY_PATH"  
      setenv LD_LIBRARY_PATH ${LMF_HOME}/lib
    endif
  else
    echo "WARNING: there is no ${LMF_HOME}/lib directory"
    echo "         you should check that LMF shared libraries are included in LD_LIBRARY_PATH"
  endif
endif
if ( ${?GATE_USE_LMF} ) then
  echo "LMF data output enabled"
else
  echo "LMF data output disabled"
endif


# We check wheter the ECAT7 home-directory variable has been set
# If that's the case, we assume that the ECAT7 library is installed on this
# system and enable the ECAT7 data output
if ( ${?ECAT7_HOME} ) then
  if ( -d $ECAT7_HOME ) then
    # Enable the use of the ECAT7 output file
    setenv GATE_USE_ECAT7 1
  else
    echo "WARNING: the ECAT7 home directory $ECAT7_HOME does not exist in your system"  
    if ( ${?GATE_USE_ECAT7} ) then
      unsetenv GATE_USE_ECAT7
    endif
  endif
endif  
if ( ${?GATE_USE_ECAT7} ) then
  echo "ECAT7 data output enabled"
else
  echo "ECAT7 data output disabled"
endif

# Enable the use of ntuples 
# This line probably is obsolete, as this variable does not seem to be used anywhere
setenv G4ANALYSIS_USE_NTUPLE 1

# $GATEHOME
# By setting this variable, Gate can be run from any directory
if ( ${?GATEHOME} ) then
  echo "GATEHOME is set to $GATEHOME"
else
  setenv GATEHOME $PWD
  echo "WARNING: variable GATEHOME is not set"
  echo "         it will be assumed to be the current directory"
  echo "         ==> " $GATEHOME
endif
if ( -d ${GATEHOME} ) then
    echo "Checking your path variable..."
    echo $PATH | grep ${GATEHOME}/bin/${G4SYSTEM} > /dev/null
    if ( $status ) then
      echo " ==> OK: ${GATEHOME}/bin/${G4SYSTEM} added to your path variable"
      setenv PATH ${GATEHOME}/bin/${G4SYSTEM}:$PATH
    else
      echo " ==> OK: ${GATEHOME}/bin/${G4SYSTEM} is already in your path variable"
    endif
    echo "Checking your LD_LIBRARY_PATH variable..."
    echo $LD_LIBRARY_PATH | grep ${GATEHOME}/tmp/${G4SYSTEM}/Gate > /dev/null
    if ( $status ) then
      echo " ==> OK: ${GATEHOME}/tmp/$G4SYSTEM/Gate added to your path variable"
      setenv LD_LIBRARY_PATH ${GATEHOME}/tmp/${G4SYSTEM}/Gate:$LD_LIBRARY_PATH
    else
      echo " ==> OK: ${GATEHOME}/tmp/${G4SYSTEM}/Gate is already in your path variable"
    endif

else
    echo "WARNING: your GATEHOME variable refers to a non-existent directory"
    unsetenv GATEHOME
endif


#########################
############## G4VERSION
#########################
if ( ${G4VERSION} == "9.3" ) then 
  setenv G4VERSION9_3 1
  unsetenv G4VERSION9_2
else if ( ${G4VERSION} == "9.4" ) then 
  setenv G4VERSION9_3 1
  unsetenv G4VERSION9_2
else
unsetenv G4VERSION9_3
endif

# NB: to switch from one G4 version to the other, it is suggested to 
# - open a new terminal (to avoid problems with former environment variable settings)
# - set the G4INSTALL and G4VERSION variables to the new one
# - source env_gate.sh (this should source also the new G4 env.sh)
# - recompile the complete GATE package
if ( ${?G4VERSION9_3} ) then
  echo "G4VERSION9_3 is set: GATE is compatible with the GEANT4 version 9.3 and 9.4"
else
  echo "G4VERSION9_3 is not set"
  echo "This GATE version is not compatible with any GEANT4 version !!!"
endif

########### Cluster Tools environment variables
#
setenv GC_DOT_GATE_DIR '.'
setenv GC_GATE_EXE_DIR $GATEHOME/bin/$G4SYSTEM
#
###############################################

echo ""
echo "Done"
echo ""
