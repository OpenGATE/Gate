/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*!
  \class  GateHounsfieldToMaterialsBuilder.cc
  \brief
  \author david.sarrut@creatis.insa-lyon.fr
*/

#include "GateHounsfieldToMaterialsBuilder.hh"
#include "GateHounsfieldMaterialTable.hh"
#include "GateHounsfieldDensityTable.hh"

//-------------------------------------------------------------------------------------------------
GateHounsfieldToMaterialsBuilder::GateHounsfieldToMaterialsBuilder()
    : mDensityTol(0.1 * g / cm3) {
    pMessenger = new GateHounsfieldToMaterialsBuilderMessenger(this);
    mMaterialTableFilename = "undefined_mMaterialTableFilename";
    mDensityTableFilename = "undefined_mDensityTableFilename";
    mOutputMaterialDatabaseFilename = "undefined_mOutputMaterialDatabaseFilename";
    mOutputHUMaterialFilename = "undefined_mOutputHUMaterialFilename";
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateHounsfieldToMaterialsBuilder::~GateHounsfieldToMaterialsBuilder() {
    delete pMessenger;
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateHounsfieldToMaterialsBuilder::BuildAndWriteMaterials() {
    GateMessage("Geometry", 3, "GateHounsfieldToMaterialsBuilder::BuildAndWriteMaterials\n");

    // Read matTable.txt
    std::vector < GateHounsfieldMaterialProperties * > mHounsfieldMaterialPropertiesVector;
    std::ifstream is;
    OpenFileInput(mMaterialTableFilename, is);
    skipComment(is);
    std::vector <G4String> elements;
    if (is) {
        G4String e;
        is >> e;
        if (e != "[Elements]") {
            GateError("The file " << mMaterialTableFilename << " must begin with [Elements]\n");
        }
        while (e != "[/Elements]") {
            is >> e;
            if (e != "[/Elements]") elements.push_back(e);
        }
    }

    while (is) {
        GateHounsfieldMaterialProperties *p = new GateHounsfieldMaterialProperties();
        p->Read(is, elements);
        if (is) mHounsfieldMaterialPropertiesVector.push_back(p);
        else delete p;
    }

    //  DD(mHounsfieldMaterialPropertiesVector.size());

    if (mHounsfieldMaterialPropertiesVector.size() < 2) {
        GateError("I manage to read " << mHounsfieldMaterialPropertiesVector.size()
                                      << " materials in the file " << mMaterialTableFilename
                                      << ". Please check it.\n");
    }

    // Read densities.txt
    GateHounsfieldDensityTable *mDensityTable = new GateHounsfieldDensityTable();
    mDensityTable->Read(mDensityTableFilename);

    // Density tolerance
    double dTol = mDensityTol;

    // Declare result
    GateHounsfieldMaterialTable *mHounsfieldMaterialTable = new GateHounsfieldMaterialTable();

    // Loop on material intervals
    for (unsigned int i = 0; i < mHounsfieldMaterialPropertiesVector.size(); i++) {
        GateMessage("Geometry", 4,
                    "Material " << i << " = " << mHounsfieldMaterialPropertiesVector[i]->GetName() << Gateendl);

        double HMin = mHounsfieldMaterialPropertiesVector[i]->GetH();
        double HMax;
        if (i == mHounsfieldMaterialPropertiesVector.size() - 1) HMax = HMin + 1;
        else HMax = mHounsfieldMaterialPropertiesVector[i + 1]->GetH();

        // Check
        if (HMax <= HMin)
            GateError("Hounsfield shoud be given in ascending order, but I read H["
                          << i << "] = " << HMin
                          << " and H[" << i + 1 << "] = " << HMax << Gateendl);
        // GateMessage("Core", 0, "H " << HMin << " " << HMax << Gateendl);

        // Find densities interval (because densities not always increase)
        double dMin = mDensityTable->GetDensityFromH(HMin);
        double dMax = mDensityTable->GetDensityFromH(HMax);
        // GateMessage("Core", 0, "Density " << dMin << " " << dMax << Gateendl);
        //     GateMessage("Core", 0, "Density " << dMin*g/cm3 << " " << dMax*g/cm3 << Gateendl);
        double dDiffMax = mDensityTable->FindMaxDensityDifference(HMin, HMax);

        double n = std::max(1., dDiffMax / dTol);
        double nNaive = std::max(1., (dMax - dMin) / dTol);
        G4String alert = (n == nNaive) ? "" : G4String(" ***** ");
        GateMessage("Core", 2, alert << "i=" << i
                                     << " (HMin,dMin)=(" << HMin << "," << G4BestUnit(dMin, "Volumic Mass") << "),"
                                     << " (HMax,dMax)=(" << HMax << "," << G4BestUnit(dMax, "Volumic Mass") << "),"
                                     << " dDiffMax=" << G4BestUnit(dDiffMax, "Volumic Mass") << ","
                                     << " n = " << n
                                     << " nNaive = " << nNaive
                                     << alert << Gateendl);

        //If material is Air divide into only one range
        if (mHounsfieldMaterialPropertiesVector[i]->GetName() == "Air") n = 1;
        if (mHounsfieldMaterialPropertiesVector[i]->GetName() == "G4_AIR") n = 1;

        double HTol = (HMax - HMin) / n;
        // GateMessage("Core", 0, "HTol = " << HTol << Gateendl);

        if (n > 1) {
            GateMessage("Geometry", 4, "Material " << mHounsfieldMaterialPropertiesVector[i]->GetName()
                                                   << " devided into " << n << " materials\n");
        }

        if (n < 0) {
            GateError("ERROR Material " << mHounsfieldMaterialPropertiesVector[i]->GetName()
                                        << " devided into " << n << " materials : density decrease from "
                                        << G4BestUnit(dMin, "Volumic Mass") << " to "
                                        << G4BestUnit(dMax, "Volumic Mass") << Gateendl);
        }

        // Loop on density interval
        for (int j = 0; j < n; j++) {
            double h1 = HMin + j * HTol;
            double h2 = std::min(HMin + (j + 1) * HTol, HMax);
            //If material is Air get the lowest density value to avoid adding importance to it
            double d = mHounsfieldMaterialPropertiesVector[i]->GetName() == "Air" ||
                       mHounsfieldMaterialPropertiesVector[i]->GetName() == "G4_AIR" ?
                       mDensityTable->GetDensityFromH(h1) :
                       mDensityTable->GetDensityFromH(h1 + (h2 - h1) / 2.0);
            GateMessage("Geometry", 4, "H1/H2 " << h1 << " " << h2 << " = "
                                                << mHounsfieldMaterialPropertiesVector[i]->GetName()
                                                << " d=" << G4BestUnit(d, "Volumic Mass") << Gateendl);
            mHounsfieldMaterialTable->AddMaterial(h1, h2, d, mHounsfieldMaterialPropertiesVector[i]);
        }
    }

    // Write final list of material
    mHounsfieldMaterialTable->WriteMaterialDatabase(mOutputMaterialDatabaseFilename);
    mHounsfieldMaterialTable->WriteMaterialtoHounsfieldLink(mOutputHUMaterialFilename);
    GateMessage("Geometry", 1, "Generation of "
        << mHounsfieldMaterialTable->GetNumberOfMaterials()
        << " materials.\n");

    // Release memory
    delete mHounsfieldMaterialTable;
    delete mDensityTable;
    elements.clear();
    for (std::vector<GateHounsfieldMaterialProperties *>::iterator it = mHounsfieldMaterialPropertiesVector.begin();
         it != mHounsfieldMaterialPropertiesVector.end();) {
        delete (*it);
        it = mHounsfieldMaterialPropertiesVector.erase(it);
    }
    if (is) is.close();
}
//-------------------------------------------------------------------------------------------------
