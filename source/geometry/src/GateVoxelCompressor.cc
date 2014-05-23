/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVoxelCompressor.hh"
#include "G4Material.hh"
#include "GateGeometryVoxelArrayStore.hh"
//E#include "GateVVolume.hh"


//-----------------------------------------------------------------------------
// Constructor
GateVoxelCompressor::GateVoxelCompressor(GateGeometryVoxelArrayStore* s):
  m_array(s),
  m_voxelSet(0),
  m_exclusionList(  new std::set<unsigned short int> ),
  m_messenger( new GateVoxelCompressorMessenger(this) ){
  // std::cout << "GateVoxelCompressor::GateVoxelCompressor - Entered." << std::endl;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Destructor
GateVoxelCompressor::~GateVoxelCompressor(){
  delete m_messenger;
  delete m_voxelSet;
  delete m_exclusionList;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4String GateVoxelCompressor::GetObjectName() const {
  return m_array->GetCreator()->GetObjectName();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelCompressor::MakeExclusionList(G4String val){

  size_t lastPos(0),curPos(0);
  while ( std::string::npos != (curPos=val.find(' ',curPos)) ){
    AddMaterial(  val.substr(lastPos,curPos-lastPos) );
    lastPos=++curPos;
  }
  AddMaterial(  val.substr(lastPos,curPos-lastPos) );
	 
  for(std::set<unsigned short int>::iterator it=m_exclusionList->begin(); it!=m_exclusionList->end(); it++) {
    G4cout << (*it) <<  std::endl;
  }

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelCompressor::AddMaterial(G4String m){
    G4Material* materialPtr =  G4Material::GetMaterial( m );
    if (materialPtr) m_exclusionList->insert( materialPtr->GetIndex() );
    else G4cout << "GateVoxelCompressor::MakeExclusionList - ERROR ! Material " << m << " not found."<< G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateVoxelCompressor::GetCompressionRatio() const{
  return  100.0 * ( 1.0 - double(m_voxelSet->size())/double(m_array->GetVoxelNx() * m_array->GetVoxelNy() * m_array->GetVoxelNz()) );
}
//-----------------------------------------------------------------------------

void GateVoxelCompressor::Initialize()
{
       if ( m_voxelSet != 0 )
          {G4cout << " %%%%%%%%%%%%%%%%   GateVoxelCompressor::Initialize() " << m_voxelSet->size() << G4endl;
           m_voxelSet->clear();
           G4cout << " %%%%%%%%%%%%%%%%   GateVoxelCompressor::Initialize() NOW SIZE IS  " << m_voxelSet->size() << G4endl;
           delete m_voxelSet;
           m_voxelSet = 0;
          }

 m_exclusionList->clear();

} /* PY Descourt 08/09/2009 */
//-----------------------------------------------------------------------------
//                                            -  Compress  -
// Note : X1 == slowest varying dimension (z)
//        X2 == ...
//        X3 == fastest varying dimension (x)
// Compression is performed in three passes: first along x, then along y and z.
void GateVoxelCompressor::Compress(){
  
        // Allocate a voxel set for the first pass
        voxelSet& voxelSetPass1(*new voxelSet);
	if (!&voxelSetPass1)
	  std::cerr << "GateVoxelCompressor::Compress - Insufficient memory for voxel set"<<std::endl<<std::flush;
	int voxelEstimate ( m_array->GetVoxelNx() * m_array->GetVoxelNy() * m_array->GetVoxelNz() );
	voxelSetPass1.reserve(voxelEstimate);
	
	// First pass - run length along X3 ( or x,  the direction varying the most rapidly)
	for(int i=0; i<m_array->GetVoxelNz() ; i++){
	  for(int j=0; j< m_array->GetVoxelNy();  j++)
	    runLength(i, j, voxelSetPass1);
	}


	// Second pass - run length along X2 (or y)
	
	//       a) sort with  minor dimension as X2 (along y). X2 is index 1.
	
	sort(voxelSetPass1.begin(), voxelSetPass1.end(),  GateCompressedVoxelOrdering(0,2,1)); // ordering( major, ..., minor)
	
	//       b) the valarray<> is the expected difference for adjacent voxels
	//          in comparison.  The last parameter (4) means that if two voxels
	//          are to be merged, their dimension at index 4 (dx2 or length in y)
	//          are to be added
	
	unsigned short int passTwo[]={0,1,0};
	voxelSet& voxelSetPass2 = runLength2nd(voxelSetPass1, std::valarray<unsigned short int>(passTwo,3), 4);
	delete &voxelSetPass1;
	
	// Third pass - run length along X1 (or z)
	sort(voxelSetPass2.begin(), voxelSetPass2.end(),  GateCompressedVoxelOrdering(1,2,0)); // ordering( major, ..., minor)
	unsigned short int passThree[]={1,0,0}; 
	m_voxelSet  = & runLength2nd(voxelSetPass2, std::valarray<unsigned short int>(passThree,3), 3);
	delete &voxelSetPass2;
	
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Initial (first pass) of run length encoding: from the matrix to a voxel set
void GateVoxelCompressor::runLength(int x1, int x2, voxelSet& vs){
  int runLength(1);
  int position (0);
  
  for (int i=0; i < m_array->GetVoxelNx()-1; i++){
    unsigned short int  value( m_array->GetVoxelMaterial(i, x2, x1)->GetIndex() );
   
    // If same material voxel, increase run length
    if ( value ==  m_array->GetVoxelMaterial(i+1, x2, x1)->GetIndex() && isNotExcluded(value) ){
      runLength++;
    }
    
    // else put out that voxel
    else{
      vs.push_back(  GateCompressedVoxel(x1, x2, position, 1, 1, runLength,  m_array->GetVoxelMaterial(position, x2, x1)->GetIndex() ) );
      position=i+1;
      runLength=1;
   }
  }
  vs.push_back(  GateCompressedVoxel(x1, x2, position, 1, 1, runLength,  m_array->GetVoxelMaterial(position, x2, x1)->GetIndex() ) );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Subsequent passes (2nd and 3rd) of run length: from a voxel set to another voxel set
// Input
//  vs     : the current voxel set
//  diff   : the expected difference in position for adjacent voxels
//  fusion : an index in GateCompressedVoxel representing one of the dimensions (dx1, dx2 or dx3) that is to be fused
voxelSet& GateVoxelCompressor::runLength2nd(voxelSet& vs, const std::valarray<unsigned short int>& diff, int fusion){
  voxelSet& newVoxels( *new voxelSet() );
  newVoxels.reserve( vs.size() );
  
  if (!&newVoxels) {
    std::cerr <<  "GateVoxelCompressor::runLength2nd - Insufficient memory for new voxel set"<<std::endl<<std::flush;
  }
  
  //  These are the indices in GateCompressedVoxel used for comparison (3:dx1, 4:dx2, 5:dx3, 6:value)
  unsigned short int array[]={3,4,5,6};
  std::valarray<unsigned short int> sid(array,4);
  
  // Initial values for the first voxel
  int runLength(1);
  int position (0);
  
  for (unsigned int i=0; i<vs.size()-1; i++){
    
    std::valarray<unsigned short int> difference(  vs[i+1].positionDifference(vs[i])  );
 
    // Compare adjacent voxels on their size and value (indices 3,4,5,6) and their coordinates (diff)
    // If equal, increase run length
    if( vs[i].compare(vs[i+1], sid) && (difference == diff)  && isNotExcluded( vs[i][6] ) ){
      runLength++;
    }

    // else output this new voxel
    else{
      GateCompressedVoxel v( vs[position] );
      v[fusion]=runLength;
      newVoxels.push_back( v );
      position=i+1;
      runLength=1;

    }
  }
  GateCompressedVoxel v( vs[position] );
  v[fusion]=runLength;
  newVoxels.push_back( v );

  return newVoxels;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool operator == (const std::valarray<unsigned short int>& lhs, const std::valarray<unsigned short int>& rhs){

  return ( lhs[0]==rhs[0] && lhs[1]==rhs[1] && lhs[2]==rhs[2] );

}
//-----------------------------------------------------------------------------
    
  
  
