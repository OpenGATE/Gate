/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATELISTOFHADMODEL_CC
#define GATELISTOFHADMODEL_CC

#include "GateListOfHadronicModels.hh"
#include "GateMessageManager.hh"

//-----------------------------------------------------------------------------
GateListOfHadronicModels::GateListOfHadronicModels(G4String model)
{
  modelName = model;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateListOfHadronicModels::~GateListOfHadronicModels()
{
  theListOfOptions.clear();
  theListOfEmin.clear();
  theListOfEmax.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateListOfHadronicModels::SetEmin(double val, G4String opt)
{

  if(opt != "NoOption")
  {
    if(!GetMaterial(opt) && !GetElement(opt))
    {
       G4cout<< "\n  <!> *** Warning *** <!> Unknown material or element: "<<opt<< Gateendl;
       return;
    }
  }

  bool set = false;
  
  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
    if(theListOfOptions[j]==opt)
    {
       theListOfEmin[j] = val;
       set = true;
    }

  if(!set)
  {
    theListOfOptions.push_back(opt);
    theListOfEmin.push_back(val);
    theListOfEmax.push_back(-1.0);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateListOfHadronicModels::SetEmax(double val, G4String opt)
{
  if(opt != "NoOption")
  {
    if(!GetMaterial(opt) && !GetElement(opt))
    {
       G4cout<< "\n  <!> *** Warning *** <!> Unknown material or element: "<<opt<< Gateendl;
       return;
    }
  }

  bool set = false;
  
  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
    if(theListOfOptions[j]==opt)
    {
       theListOfEmax[j] = val;
       set = true;
    }

  if(!set)
  {
    theListOfOptions.push_back(opt);
    theListOfEmax.push_back(val);
    theListOfEmin.push_back(-1.0);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4double GateListOfHadronicModels::GetEmin( G4String opt)
{
  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
    if(opt == theListOfOptions[j]) return theListOfEmin[j];
  
  return -1.;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4double GateListOfHadronicModels::GetEmax( G4String opt)
{
  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
    if(opt == theListOfOptions[j]) return theListOfEmax[j];
  
  return -1.;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateListOfHadronicModels::ClearERange()
{
  theListOfEmin.clear();
  theListOfEmax.clear();
  theListOfOptions.clear();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Material * GateListOfHadronicModels::GetMaterial(G4String materialName)
{
   
  const G4MaterialTable* matTbl = G4Material::GetMaterialTable();
  
  for(G4MaterialTable::const_iterator it=matTbl->begin();it!=matTbl->end();++it)
  {
    if((*it)->GetName() == materialName) return (*it);
  }
  return 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4Element * GateListOfHadronicModels::GetElement(G4String elementName)
{
   
  const G4ElementTable * elemTbl  = G4Element::GetElementTable();
  
  for(G4ElementTable::const_iterator it=elemTbl->begin();it!=elemTbl->end();++it)
  {
    if((*it)->GetName() == elementName) return (*it);
  }
  return 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateListOfHadronicModels::Print(G4int level, G4String symbol, G4String symbol2)
{
  G4String space = " ";
  for(G4int i = 1;i<level;++i) space += "  ";
  G4String space2 = space + "  " + symbol2 + " ";
  space += symbol + " ";

  std::cout<<space<<modelName << Gateendl;


  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
  {
    if(theListOfOptions[j]=="NoOption")
    {
      std::cout<<space2<<std::flush;
      if(theListOfEmin[j]>0) std::cout<<"Emin = "<<std::setw(3)<<G4BestUnit(theListOfEmin[j],"Energy")<<std::flush;

      if(theListOfEmin[j]>0 && theListOfEmax[j]>0) std::cout<<"  -->  "<<std::flush;

      if(theListOfEmax[j]>0) std::cout<<"Emax = "<<std::setw(3)<<G4BestUnit(theListOfEmax[j],"Energy")<<std::flush;

      std::cout<< Gateendl;
    }
  }

  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
  {
    if(theListOfOptions[j]!="NoOption")
    {
      std::cout<<space2<<std::flush;

      if(theListOfEmin[j]>0) std::cout<<"Emin = "<<std::setw(3)<<G4BestUnit(theListOfEmin[j],"Energy")<<std::flush;

      if(theListOfEmin[j]>0 && theListOfEmax[j]>0) std::cout<<"  -->  "<<std::flush;

      if(theListOfEmax[j]>0) std::cout<<"Emax = "<<std::setw(3)<<G4BestUnit(theListOfEmax[j],"Energy")<<std::flush;

      std::cout<<" ("<<theListOfOptions[j]<<")\n";
    }
  }     

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateListOfHadronicModels::Print(G4String file,G4int level, G4String symbol, G4String symbol2)
{
  std::ofstream os;
  os.open(file.data(), std::ios_base::app);

  G4String space = " ";
  for(G4int i = 1;i<level;++i) space += "  ";
  G4String space2 = space + "  " + symbol2 + " ";
  space += symbol + " ";

  os<<space<<modelName.data() << Gateendl;


  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
  {
    if(theListOfOptions[j]=="NoOption")
    {
      os<<space2<<std::flush;
      if(theListOfEmin[j]>0) os<<"Emin = "<<std::setw(3)<<G4BestUnit(theListOfEmin[j],"Energy")<<std::flush;

      if(theListOfEmin[j]>0 && theListOfEmax[j]>0) os<<"  -->  "<<std::flush;

      if(theListOfEmax[j]>0) os<<"Emax = "<<std::setw(3)<<G4BestUnit(theListOfEmax[j],"Energy")<<std::flush;

      os<< Gateendl;
    }
  }

  for(unsigned int j=0; j<theListOfOptions.size(); ++j)
  {
    if(theListOfOptions[j]!="NoOption")
    {
      os<<space2<<std::flush;

      if(theListOfEmin[j]>0) os<<"Emin = "<<std::setw(3)<<G4BestUnit(theListOfEmin[j],"Energy")<<std::flush;

      if(theListOfEmin[j]>0 && theListOfEmax[j]>0) os<<"  -->  "<<std::flush;

      if(theListOfEmax[j]>0) os<<"Emax = "<<std::setw(3)<<G4BestUnit(theListOfEmax[j],"Energy")<<std::flush;

      os<<" ("<<theListOfOptions[j].data()<<")\n";
    }
  }     
  os.close();
}
//-----------------------------------------------------------------------------


#endif
