/*
 * GateVoxelizedPosDistribution.cc
 *
 *  Created on: 20 nov. 2015
 *      Author: js244228
 */

#include "GateVoxelizedPosDistribution.hh"
#include "Randomize.hh"

// trim, tokenize, and get_key_index really belong elsewhere since they're
// general functions for parsing the header file

G4String trim(const G4String& str,  const G4String& whitespace = " \t")
{
    std::size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == G4String::npos)
        return ""; // no content

    std::size_t strEnd = str.find_last_not_of(whitespace);
    return str.substr(strBegin, strEnd - strBegin + 1);
}

bool tokenize(G4String str, G4String &key, G4String &value)
{
  // split at :=
  std::size_t separator;
  std::size_t start=0;
  std::size_t end;

  // strip off comments
  end = str.find(";");
  str = str.substr(0,end);

  // locate separator
  separator = str.find(":=");
  if(separator==str.npos)
    return false;

  key = str.substr(start,separator);
  key = trim(key, " \t_!");             // trim whitespace, including <underscore> and <!>
  value = str.substr(separator+2);
  value = trim(value);                  // trim whitespace
  return true;
}

G4int get_key_index(const G4String str)
{
  G4int index;
  std::size_t pos;
  pos = str.find("[");
  if(pos==str.npos)
    return -1;
  index = std::atoi(str.substr(pos+1).c_str());
  return index;
}

G4String get_path(const G4String str)
{
  std::size_t found = str.find_last_of("/\\");
  if(found==str.npos)
    return "";
  return str.substr(0,found+1);
}

GateVoxelizedPosDistribution::GateVoxelizedPosDistribution(G4String filename)
{
  G4int i, j, k;

  G4String data_filename;
  std::ifstream f_in;
  G4String buffer;
  G4String key;
  G4String value;

  G4cout << "GateVoxelizedPosDistribution(" << "\"" << filename << "\")" << G4endl;

  mPosition.set(0.0,0.0,0.0);
  mResolution.set(0.0,0.0,0.0);
  m_nx = 0;
  m_ny = 0;
  m_nz = 0;

  // read header file
  f_in.open(filename);
  if(!f_in.good())
  {
    G4cout << "Unable to open voxelized data file:" << filename << G4endl;
    return;
  }

  while(!f_in.eof())
  {
    std::getline(f_in, buffer);
    if(!tokenize(buffer,key,value))
      continue;
    G4cout << "key: " << key << "\t value: " << value << G4endl;

    if(key.find("name of data file")==0)
      data_filename = get_path(filename) + value;
    else if(key.find("matrix size")==0)
    {
      i = get_key_index(key);
      switch(i)
      {
      case 1:
        m_nx = std::atoi(value.c_str());
        break;
      case 2:
        m_ny = std::atoi(value.c_str());
        break;
      case 3:
        m_nz = std::atoi(value.c_str());
        break;
      default:
        G4cout << "Invalid matrix size index." << G4endl;
      }
    }
    else if(key.find("scale factor")==0)
    {
      i = get_key_index(key) - 1;
      if(i<3 && i>= 0)
        mResolution[i] = std::atof(value.c_str()) * CLHEP::mm;
      else
        G4cout << "Invalid scale factor index." << G4endl;
    }

  }
  f_in.close();

  // flip y_axis direction
  mResolution[1] = -mResolution[1];

  mPosDistZCDF = new G4double[m_nz];
  mPosDistYCDF = new G4double*[m_nz];
  mPosDistXCDF = new G4double**[m_nz];
  for(i=0;i<m_nz;i++)
  {
    mPosDistYCDF[i] = new G4double[m_ny];
    mPosDistXCDF[i] = new G4double*[m_ny];
  }
  for(i=0;i<m_nz;i++)
    for(j=0;j<m_ny;j++)
      mPosDistXCDF[i][j] = new G4double[m_nx];

  G4float *temp = new G4float[m_nx];

  f_in.open(data_filename, std::ios::binary);
  if(!f_in.good())
    G4cout << "Error opening data file: " << data_filename << G4endl;

  for(i=0;i<m_nz;i++)
  {
    for(j=0;j<m_ny;j++)
    {
      f_in.read(reinterpret_cast<char*>(temp),m_nx*sizeof(G4float));
      mPosDistXCDF[i][j][0] = temp[0];
      for(k=1;k<m_nx;k++)
        mPosDistXCDF[i][j][k] = mPosDistXCDF[i][j][k-1] + temp[k];

      // copy sum to y distribution
      if(j==0)
        mPosDistYCDF[i][j] = mPosDistXCDF[i][j][m_nx-1];
      else
        mPosDistYCDF[i][j] = mPosDistYCDF[i][j-1]+ mPosDistXCDF[i][j][m_nx-1];

      // normalize cumulative x distribution
      if(mPosDistXCDF[i][j][m_nx-1] != 0.0)
        for(k=0;k<m_nx;k++)
          mPosDistXCDF[i][j][k] /= mPosDistXCDF[i][j][m_nx-1];

    }
    // copy sum to z distribution
    if (i==0)
      mPosDistZCDF[i] = mPosDistYCDF[i][m_ny-1];
    else
      mPosDistZCDF[i] = mPosDistZCDF[i-1] + mPosDistYCDF[i][m_ny-1];

    // normalize cumulative y distribution
    if(mPosDistYCDF[i][m_ny-1] != 0.0)
      for(j=0;j<m_ny;j++)
        mPosDistYCDF[i][j] /= mPosDistYCDF[i][m_ny-1];
  }
  // normalize cumulative z distribution
  if(mPosDistZCDF[m_nz-1] != 0.0)
    for(i=0;i<m_nz;i++)
      mPosDistZCDF[i] /= mPosDistZCDF[m_nz-1];

  f_in.close();
  delete [] temp;

  mPosition.set(-(m_nx/2)*mResolution[0],-(m_ny/2)*mResolution[1],-(m_nz/2)*mResolution[2]);

  G4cout << data_filename << G4endl;
  G4cout << "(" << m_nx << "," << m_ny << "," << m_nz << ")" << G4endl;
  G4cout << mResolution << G4endl;
  G4cout << mPosition << G4endl;

}

GateVoxelizedPosDistribution::~GateVoxelizedPosDistribution()
{
  int i,j;

  for(i=0;i<m_nz;i++)
  {
    for(j=0;j<m_ny;j++)
      if(mPosDistXCDF[i][j])
        delete [] mPosDistXCDF[i][j];

    if(mPosDistXCDF[i])
      delete [] mPosDistXCDF[i];

    if(mPosDistYCDF[i])
      delete [] mPosDistYCDF[i];
  }

  if(mPosDistXCDF)
    delete [] mPosDistXCDF;
  if(mPosDistYCDF)
    delete [] mPosDistYCDF;
  if(mPosDistZCDF)
    delete [] mPosDistZCDF;
}

G4ThreeVector GateVoxelizedPosDistribution::GenerateOne()
{
  G4int i, j, k;
  G4double p;
  G4ThreeVector pos;

  i=0;
  p = G4UniformRand();
  while(p > mPosDistZCDF[i])
    i++;

  j=0;
  p=G4UniformRand();
  while(p > mPosDistYCDF[i][j])
    j++;

  k=0;
  p=G4UniformRand();
  while(p > mPosDistXCDF[i][j][k])
    k++;

  pos.set(mResolution[0] * (k + G4UniformRand()),
          mResolution[1] * (j + G4UniformRand()),
          mResolution[2] * (i + G4UniformRand()) );
  pos += mPosition;

  return pos;

}
