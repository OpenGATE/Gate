/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEREGULARPARAMETERIZEDINSERTER_HH
#define GATEREGULARPARAMETERIZEDINSERTER_HH 1

#include "globals.hh"
#include "GateBox.hh"
#include "GateVGeometryVoxelReader.hh"
#include "GateRegularParameterization.hh"
#include "GateRegularParam.hh"
#include "GateRegularParameterizedMessenger.hh"

class GateRegularParameterized : public GateBox
{
public:
    //! Constructor1
    GateRegularParameterized(const G4String& name,
    				     G4bool acceptsChildren=true, 
		 		     G4int depth=0);

    //! Constructor2
    GateRegularParameterized(const G4String& name);

    //! Destructor
    ~GateRegularParameterized();

    FCT_FOR_AUTO_CREATOR_VOLUME(GateRegularParameterized)

    //! Insert the reader to read and store the voxel material info
    void InsertReader(G4String readerType);

    //! Remove the reader to read and store the voxel material info
    void RemoveReader();

    //! Attach the matrix to the phantom SD
    void AttachPhantomSD();

    //! Attach an new output module for dose calculation
    void AddOutput(G4String name);

    //! Implementation of virtual method ConstructGeometry
    void ConstructGeometry(G4LogicalVolume* mother_log, G4bool flagUpdateOnly);

    /*! Implementation of virtual method ConstructOwnPhysicalVolumes.
        It is in fact a copy of the AutoplacedCreatorInserter same method
        but instead of placing a new physical volume (that creates one too),
        we take the physical container created and used by the regular
        parameterization. !*/
    void ConstructOwnPhysicalVolume(G4bool flagUpdateOnly);

    //! Get the reader to read and store the voxel material info
    inline GateVGeometryVoxelReader* GetReader() const
      { return m_voxelReader;}

    //! Get and Set the verbose level
    inline G4int GetVerbosity ()                      {return verboseLevel;}
    inline void  SetVerbosity (G4int theVerboseLevel) {verboseLevel=theVerboseLevel;}

    //! Get the name m_name
    inline G4String GetName() {return m_name;}

    //! Get and Set the SkipEqualMaterials parameter
    inline void ChangeSkipEqualMaterials(G4bool aBool) {if (aBool) skipEqualMaterials=1; else skipEqualMaterials=0;}
    inline G4int GetSkipEqualMaterials() {return skipEqualMaterials;}

private:

    G4String m_name;
    G4int verboseLevel;
    G4int skipEqualMaterials;

    GateRegularParameterizedMessenger*  m_messenger;
    GateVGeometryVoxelReader*           m_voxelReader;
    GateRegularParam*                   m_voxelInserter;

    G4ThreeVector                       voxelNumber;
    G4ThreeVector                       voxelSize;
};

MAKE_AUTO_CREATOR_VOLUME(regularMatrix,GateRegularParameterized)

#endif
