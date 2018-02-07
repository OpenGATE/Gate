/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#include "GateConfiguration.h"
#include "GateMessageManager.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateARFTableMgr.hh"
#include "GateARFTable.hh"
#include "GateARFTableMgrMessenger.hh"
#include "globals.hh"
#include "G4ios.hh"
#include <map>
#include <utility>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "GateARFSD.hh"

GateARFTableMgr::GateARFTableMgr(const G4String & aName, GateARFSD* arfSD)
  {
  mTableName = aName;
  mMessenger = new GateARFTableMgrMessenger(aName, this);
  mEnergyThreshHold = 0.;
  mEnergyUpHold = 0.;
  mArfSD = arfSD;
  mDistance = 34.6 * cm;
  mSaveArfTables = 0;
  mLoadArfTables = 0;
  mBinaryFilename = G4String("ARFTables.bin");
  mCurrentIndex = 0;
  mNumberOfBins = 100;
  }

GateARFTableMgr::~GateARFTableMgr()
  {
  delete mMessenger;
  }

G4double GateARFTableMgr::ScanTables(const G4double & x,
                                     const G4double & y,
                                     const G4double & energy)
  {
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    if ((energy - ((*mapIterator).second)->GetElow() > 1.e-8)
        && (energy - ((*mapIterator).second)->GetEhigh() < 1.e-8))
      {
      return (((*mapIterator).second)->RetrieveProbability(x, y));
      break;
      }
    }
  return 0.;
  }

void GateARFTableMgr::AddaTable(GateARFTable* arfTable)
  {
  arfTable->SetIndex(mCurrentIndex);
  mArfTableMap.insert(std::make_pair(mCurrentIndex, arfTable));
  mCurrentIndex++;
  }

void GateARFTableMgr::ComputeARFTablesFromEW(const G4String & filename)
  {
  mArfDataFileTxt = filename;
  /* read the spectrum data and construct as many ARF tables as energy intervals */
  std::ifstream inputData(mArfDataFileTxt.c_str(), std::ios::in);

  if (!inputData)
    {
    G4String AString = "Cannot open file " + mArfDataFileTxt;
    G4Exception("GateARFTableMgr::ComputeARFTables", "ComputeARFTables", FatalException, AString);
    return;
    }
  char buffer[200];
  G4double minEnergy = 0;
  G4double maxEnergy = 0;
  G4int numberOfFiles = 0;
  G4String rootBaseFilename;
  G4String baseName = GetName() + "ARFTable_";

  while (!inputData.eof())
    {
    inputData.getline(buffer, 200);
    std::istringstream inputBuffer(buffer);
    inputBuffer.clear();
    inputBuffer.str(buffer);
    G4String bufferString = inputBuffer.str();
    if (bufferString != "" && (bufferString.find("#", 0) != 0) && (bufferString.find("!", 0) != 0))
      {
      inputBuffer >> minEnergy >> maxEnergy >> rootBaseFilename >> numberOfFiles;
      GetARFSD()->AddNewEnergyWindow(rootBaseFilename, numberOfFiles);
      std::ostringstream currentIndexOss;
      currentIndexOss << mCurrentIndex;
      G4String tableName = baseName + currentIndexOss.str();
      GateARFTable* arfTable = new GateARFTable(tableName);
      arfTable->SetElow(minEnergy * keV);
      arfTable->SetEhigh(maxEnergy * keV);
      arfTable->SetEnergyReso(mEnergyResolution);
      arfTable->SetERef(mEnergyOfReference);
      arfTable->SetDistanceFromSourceToDetector(mDistance);
      AddaTable(arfTable);
      }
    }
  mArfSD->computeTables();
  inputData.close();
  }

G4int GateARFTableMgr::InitializeTables()
  {

  if (mLoadArfTables == 1)
    {
    return 1;
    }
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    ((*mapIterator).second)->Initialize(mEnergyThreshHold, mEnergyUpHold);
    }
  G4cout << "GateARFTableMgr::InitializeTables() : All ARF Tables are initialized \n";
  return 0;
  }
void GateARFTableMgr::SetNSimuPhotons(G4double* NPhotons)
  {
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  G4int i = 0;
  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    ((*mapIterator).second)->SetNSimuPhotons(NPhotons[i]);
    i++;
    }
  }

void GateARFTableMgr::SetNBins(const G4int & N)
  {
  mNumberOfBins = N;
  }

void GateARFTableMgr::SetEReso(const G4double & energyResolution)
  {
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  mEnergyResolution = energyResolution;
  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    ((*mapIterator).second)->SetEnergyReso(energyResolution);
    }
  }

void GateARFTableMgr::SetERef(const G4double & energyOfReference)
  {
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  mEnergyOfReference = energyOfReference;
  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    ((*mapIterator).second)->SetERef(energyOfReference);
    }
  }

void GateARFTableMgr::convertDRF2ARF()
  {
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  G4cout << " GateARFTableMgr::convertDRF2ARF()   CONVERTING DRF tables to ARF TABLES\n";
  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    ((*mapIterator).second)->convertDRF2ARF();
    }
  }

void GateARFTableMgr::FillDRFTable(const G4int & iT,
                                   const G4double & depositedEnergy,
                                   const G4double & projectedX,
                                   const G4double & projectedY)
  {
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  mapIterator = mArfTableMap.find(iT);
  if (mapIterator != mArfTableMap.end())
    {
    ((*mapIterator).second)->FillDRFTable(depositedEnergy, projectedX, projectedY);
    }
  else
    {
    G4cout << " WARNING :: GateARFTableMgr::FillTable : Table # "
           << iT
           << " does not exist. Ignored \n";
    }
  }

void GateARFTableMgr::ListTables()
  {
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  G4cout << " ----------------------------------------------------------------------------------------------\n";
  G4cout << " GateARFTableMgr::ListTables()  List of the ARF Tables for ARF Sensitive Detector "
         << GetName()
         << Gateendl;

  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    ((*mapIterator).second)->Describe();
    }
  G4cout << " ----------------------------------------------------------------------------------------------\n";
  }

/* save the ARF tables to a binary file */

void GateARFTableMgr::SaveARFToBinaryFile()
  {
  std::ofstream outputBinaryFile;
  outputBinaryFile.open(mBinaryFilename.c_str());
  std::map<G4int, GateARFTable*>::iterator mapIterator;
  mapIterator = mArfTableMap.begin();
  outputBinaryFile.seekp(0, std::ios::beg);
  G4double tableSize = G4double(mArfTableMap.size());
  outputBinaryFile.write((const char*) (&tableSize), sizeof(G4double));
  tableSize = G4double(((*mapIterator).second)->GetTotalNb());
  outputBinaryFile.write((const char*) (&tableSize), sizeof(G4double));
  G4int bytesNumber = 8 + G4int(tableSize);
  size_t tableBufferSize = bytesNumber * sizeof(G4double);
  long writingPosition;
  for (mapIterator = mArfTableMap.begin(); mapIterator != mArfTableMap.end(); mapIterator++)
    {
    G4double* tableBuffer = new G4double[bytesNumber];
    ((*mapIterator).second)->GetARFAsBinaryBuffer(tableBuffer);
    writingPosition = outputBinaryFile.tellp();
    writingPosition++;
    outputBinaryFile.seekp(writingPosition, std::ios::beg);
    if (outputBinaryFile.bad())
      {
      G4String msg = "Could not locate the position where to write the ARFTable onto the disk (out of disk space?)";
      G4Exception("GateARFTableMgr::SaveARFToBinaryFile",
                  "SaveARFToBinaryFile",
                  FatalException,
                  msg);
      }
    G4cout << " Writing ARF Table "
           << ((*mapIterator).second)->GetName()
           << " to file "
           << mBinaryFilename
           << " at position "
           << writingPosition;
    outputBinaryFile.write((const char*) (tableBuffer), tableBufferSize);
    G4cout << "...[OK] : " << tableBufferSize << " bytes\n";
    if (outputBinaryFile.bad())
      {
      G4String msg = "Could not write the ARFTable onto the disk (out of disk space?)";
      G4Exception("GateARFTableMgr::SaveARFToBinaryFile",
                  "SaveARFToBinaryFile",
                  FatalException,
                  msg);
      }
    outputBinaryFile.flush();
    delete[] tableBuffer;
    }
  G4cout << " All ARF tables have been saved to binary file\n " << mBinaryFilename;
  outputBinaryFile.close();
  }

void GateARFTableMgr::LoadARFFromBinaryFile(const G4String & binaryFilename)
  {
  mLoadArfTables = 1;
  mCurrentIndex = 0;
  G4String basename = GetName() + "ARFTable_";
  std::ifstream inputBinaryFile;
  inputBinaryFile.open(binaryFilename.c_str(), std::ios::binary);
  inputBinaryFile.seekg(0, std::ios::beg);
  G4double nbOfTables = 0;
  G4double tableSize = 0;
  inputBinaryFile.read((char*) (&nbOfTables), sizeof(G4double));
  long readingPosition = inputBinaryFile.tellg();
  G4cout << "Nb Of Tables " << nbOfTables << Gateendl;
  G4cout << readingPosition << Gateendl;
  inputBinaryFile.seekg(readingPosition, std::ios::beg);
  inputBinaryFile.read((char*) (&tableSize), sizeof(G4double));
  G4cout << "the size " << tableSize << Gateendl;
  G4cout << readingPosition << Gateendl;

  G4int bytesNumber = 8 + int(tableSize);
  size_t tableBufferSize = bytesNumber * sizeof(G4double);
  G4double* tableBuffer = new G4double[bytesNumber];
  for (size_t i = 0; i < size_t(nbOfTables); i++)
    {
    std::ostringstream oss;
    oss << mCurrentIndex;
    G4String tableName = basename + oss.str();
    readingPosition = inputBinaryFile.tellg();
    readingPosition++;
    G4cout << " Reading ARF Table "
           << tableName
           << " from file "
           << binaryFilename
           << " at position ";

    inputBinaryFile.seekg(readingPosition, std::ios::beg);
    inputBinaryFile.read((char*) (tableBuffer), tableBufferSize);
    G4cout << readingPosition << "...[OK] : " << tableBufferSize << " bytes\n";
    GateARFTable* arfTable = new GateARFTable(tableName);
    arfTable->Initialize(tableBuffer[4], tableBuffer[5]);
    arfTable->SetEnergyReso(tableBuffer[2]);
    arfTable->SetERef(tableBuffer[3]);
    G4cout << tableBuffer[0]
           << "  "
           << tableBuffer[1]
           << "  "
           << tableBuffer[2]
           << "  "
           << tableBuffer[3]
           << "  "
           << tableBuffer[4]
           << Gateendl;
    G4cout << tableBuffer[5] << Gateendl;
    arfTable->FillTableFromBuffer(tableBuffer);
    AddaTable(arfTable);
    }
  inputBinaryFile.close();
  delete[] tableBuffer;
  ListTables();
  }

#endif

