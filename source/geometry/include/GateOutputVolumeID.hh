/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateOutputVolumeID_h
#define GateOutputVolumeID_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#define OUTPUTVOLUMEID_SIZE  6

/*! \class  GateOutputVolumeID
    \brief  Class for storing an identifier of a volume as a vector of integer
    \brief  This volume ID can be stored into output files for analysis and image reconstruction
    
    - GateOutputVolumeID - by Daniel.Strul@iphe.unil.ch
    
    - Output volume IDs are created by systems (GateVSystem) after analysis of a hit:
      Each output volume ID identifies a volume according to the geometry model defined
      by the system

      \sa GateVSystem
*/      
class GateOutputVolumeID : public std::vector<G4int>
{
  public:
    //! Constructor
    GateOutputVolumeID(size_t itsSize=OUTPUTVOLUMEID_SIZE);

    //! Destructor
    virtual ~GateOutputVolumeID() {}

    //! Friend function: inserts (prints) a GateOutputVolumeID into a stream
    friend std::ostream& operator<<(std::ostream&, const GateOutputVolumeID& volumeID);    

    //! Check whether the ID is valid, i.e. not empty and with no negative element
    G4bool IsValid() const;

    //! Check whether the ID is invalid, i.e. either empty or with at least one negative element
    inline G4bool IsInvalid() const
      { return !(IsValid()); }

    //! Extract the topmost elements of the ID, down to the level 'depth'
    //! Returns an ID with (depth+1) elements
    GateOutputVolumeID Top(size_t depth) const;
};

inline GateOutputVolumeID::GateOutputVolumeID(size_t itsSize)
 : std::vector<G4int>(itsSize, -1)
{}


#define BASE_DEPTH    	   0
#define RSECTOR_DEPTH      1
#define MODULE_DEPTH       2
#define SUBMODULE_DEPTH    3
#define CRYSTAL_DEPTH      4
#define LAYER_DEPTH    	   5

#endif
