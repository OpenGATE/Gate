/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#ifndef GateVoxelCompressor_H
#define GateVoxelCompressor_H 1

#include <set>
#include "globals.hh"
#include "GateCompressedVoxel.hh"
#include "GateVoxelCompressorMessenger.hh"

class GateGeometryVoxelArrayStore;

bool operator == (const std::valarray<unsigned short int>& lhs, const std::valarray<unsigned short int>& rhs);

class GateVoxelCompressor{
  
public:
  
  GateVoxelCompressor(GateGeometryVoxelArrayStore* s);
  ~GateVoxelCompressor();
  
    void Initialize();/* PY Descourt 08/09/2009 */  
	
  void      Compress();
  void      runLength(int x1, int x2, voxelSet& vs);
  voxelSet& runLength2nd(voxelSet& vs, const std::valarray<unsigned short int>& chk, int fusion);

  const GateCompressedVoxel& GetVoxel(int copyNo) const { return (*m_voxelSet)[copyNo]; }
  int                        GetNbOfCopies()      const { return m_voxelSet->size();    }

  double GetCompressionRatio() const;

  void   MakeExclusionList(G4String s);
  
  G4String GetObjectName() const;

  void Print(){
    for (voxelSet::iterator it=m_voxelSet->begin(); it!=m_voxelSet->end();it++) std::cout << (*it) << std::endl;
  }

private:

  void AddMaterial(G4String m);
  bool isNotExcluded(unsigned short int n) const { return  m_exclusionList->find( n )==m_exclusionList->end();}

private:
  
  GateGeometryVoxelArrayStore*   m_array;
  voxelSet*                      m_voxelSet;
  std::set<unsigned short int>*  m_exclusionList;
  GateVoxelCompressorMessenger*  m_messenger;
  
};

#endif
