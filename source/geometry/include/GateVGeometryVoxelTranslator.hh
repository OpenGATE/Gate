/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateVGeometryVoxelTranslator_h
#define GateVGeometryVoxelTranslator_h 1

#include "GateVGeometryVoxelReader.hh"
#include "globals.hh"

class GateVGeometryVoxelTranslatorMessenger;

/*! \class  GateVGeometryVoxelTranslator
    \brief  This class is used to interpret the information in the digital images as material info
    
    - GateVGeometryVoxelTranslator - by Giovanni.Santin@cern.ch
    
    - It has inside the rules to go from the image to the materials, on the base of funcions or
      tabulated information

    - The concrete translator classes have to implement at least the abstract method TranslateToMaterial 

      \sa GateGeometryVoxelTabulatedTranslator
      \sa GateVGeometryVoxelTranslatorMessenger
      \sa GateVGeometryVoxelReader
      \sa GateVSourceVoxelReader
      \sa GateVSourceVoxelTranslator
*/      

class GateVGeometryVoxelTranslator
{
public:
  GateVGeometryVoxelTranslator(GateVGeometryVoxelReader* voxelReader);
  virtual ~GateVGeometryVoxelTranslator() {};
  
public:

  virtual G4String TranslateToMaterial(G4int voxelValue) = 0;

  virtual GateVGeometryVoxelReader* GetReader() { return m_voxelReader; };
  virtual G4String                  GetName()   { return m_name; };
  
  //! Modif DS: Pure virtual method to iterate through the material's list
  virtual G4String GetNextMaterial(G4bool doReset=false) = 0 ;

  virtual G4VisAttributes* GetMaterialAttributes(G4Material* m) = 0 ;

  //! Will be used by regularParameterization
  virtual void GetCompleteListOfMaterials(std::vector<G4String>& mat) = 0;

protected:
  G4String                          m_name;
  GateVGeometryVoxelReader*         m_voxelReader;

};

#endif

