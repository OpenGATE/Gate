#ifndef GATEMATERIALMUHANDLER_CC
#define GATEMATERIALMUHANDLER_CC

#include "GateMaterialMuHandler.hh"
#include "GateMuTables.hh"
#include "GateMiscFunctions.hh"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
using std::map;
using std::string;

GateMaterialMuHandler::GateMaterialMuHandler(int nbOfElements)
{
  mElementsTable = new GateMuTable*[nbOfElements+1];
  mNbOfElements = nbOfElements;
  InitElementTable();
}


GateMaterialMuHandler::~GateMaterialMuHandler()
{
  delete[] mElementsTable;
}

inline double interpolation(double Xa,double Xb,double Ya,double Yb,double x){
  return exp(log(Ya) + log(Yb/Ya) / log(Xb/Xa)* log(x/Xa) );
}

void GateMaterialMuHandler::AddMaterial(G4Material* material)
{
  //const G4ElementVector* elements = material->GetElementVector();
  int nb_e = 0;
  int nb_of_elements = material->GetNumberOfElements();
  for(int i = 0; i < nb_of_elements; i++)
    nb_e += mElementsTable[(int) material->GetElement(i)->GetZ()]->GetSize();
  
  double* energies = new double[nb_e];
  int *index = new int[nb_of_elements];
  double **e_tables = new double*[nb_of_elements];
  double **mu_tables = new double*[nb_of_elements];
  double **muen_tables = new double*[nb_of_elements];
  //  int min_index;
  
  const G4double* FractionMass = material->GetFractionVector();

  for(int i = 0; i < nb_of_elements; i++){
    e_tables[i] = mElementsTable[(int) material->GetElement(i)->GetZ()]->GetEnergies();
    mu_tables[i] = mElementsTable[(int) material->GetElement(i)->GetZ()]->GetMuTable();
    muen_tables[i] = mElementsTable[(int) material->GetElement(i)->GetZ()]->GetMuEnTable();
    index[i] = 0;
  }
  for(int i = 0; i < nb_e; i++){
    int min_table = 0;
    while(index[min_table] >= mElementsTable[(int) material->GetElement(min_table)->GetZ()]->GetSize())
      min_table++;
    for(int j = min_table + 1; j < nb_of_elements; j++)
      if(e_tables[j][index[j]] < e_tables[min_table][index[min_table]])
  	min_table = j;
    energies[i] = e_tables[min_table][index[min_table]];
    
    if(i > 0){
      if(energies[i] == energies[i-1]){
  	if(index[min_table] > 0 && e_tables[min_table][index[min_table]] == 
  	   e_tables[min_table][index[min_table]-1])
  	  ;
  	else{
  	  i--;
  	  nb_e--;
  	}
      }
    }
    index[min_table]++;
  }
  
  //And now computing mu_en
  double *MuEn = new double[nb_e];
  double *Mu = new double[nb_e];
  for(int i = 0; i < nb_of_elements; i++){
    index[i] = 0;
  }
  

  //Assume that all table begin with the same energy
  for(int i = 0; i < nb_e; i++){
    MuEn[i] = 0.0;
    Mu[i] = 0.0;
    double current_e = energies[i];
    for(int j = 0; j < nb_of_elements; j++){
      //You never need to advance twice
      if(e_tables[j][index[j]] < current_e)
  	index[j]++;
      if(e_tables[j][index[j]] == current_e){
  	Mu[i] += FractionMass[j]*mu_tables[j][index[j]];
  	MuEn[i] += FractionMass[j]*muen_tables[j][index[j]];
	if(i != nb_e-1)
	  if(e_tables[j][index[j]] == e_tables[j][index[j]+1])
	    index[j]++;
      }
      else{
  	Mu[i] += FractionMass[j]*interpolation(e_tables[j][index[j]-1],
						 e_tables[j][index[j]],
						 mu_tables[j][index[j]-1],
						 mu_tables[j][index[j]],
						 current_e);
  	MuEn[i] += FractionMass[j]*interpolation(e_tables[j][index[j]-1],
						 e_tables[j][index[j]],
						 muen_tables[j][index[j]-1],
						 muen_tables[j][index[j]],
						 current_e);

      }
    }
  }
  
  GateMuTable * table = new GateMuTable(material->GetName(), nb_e);
  
  for(int i = 0; i < nb_e; i++){
    table->PutValue(i, energies[i], Mu[i], MuEn[i]);
  }
  
  mMaterialTable.insert(std::pair<G4String, GateMuTable*>(material->GetName(),table));
}

void GateMaterialMuHandler::ReadElementFile(int z)
{
  std::ostringstream stream;
  stream << z;
  string filenameMu = "Mu-"+ stream.str() +".dat";
  string filenameMuEn = "Muen-"+ stream.str() +".dat";
  
  std::ifstream fileMu, fileMuEn;
  fileMu.open(filenameMu.c_str());
  fileMuEn.open(filenameMuEn.c_str());
  int nblines;
  fileMu >> nblines;
  fileMuEn >> nblines;
  GateMuTable* table = new GateMuTable(string(), nblines);
  mElementsTable[z] = table;
  for(int j = 0; j < nblines; j++){
    double e, mu, muen;
    fileMu >> e >> mu;
    fileMuEn >> e >> muen;
    table->PutValue(j, e, mu, muen);
  }
  fileMu.close();
  fileMuEn.close();  
}

void GateMaterialMuHandler::InitElementTable()
{
  for(int i = 1; i <= mNbOfElements; i++)
    ReadElementFile(i);
}

double GateMaterialMuHandler::GetAttenuation(G4Material* material, double energy)
{
  map<G4String, GateMuTable*>::iterator it = mMaterialTable.find(material->GetName());
  if(it == mMaterialTable.end()){
    AddMaterial(material);
  }
  
  return mMaterialTable[material->GetName()]->GetMuEn(energy);
}

double GateMaterialMuHandler::GetMu(G4Material* material, double energy)
{
  map<G4String, GateMuTable*>::iterator it = mMaterialTable.find(material->GetName());
  if(it == mMaterialTable.end()){
    AddMaterial(material);
  }
  return mMaterialTable[material->GetName()]->GetMu(energy);
}

#endif
