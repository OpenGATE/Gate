// vim: cindent smartindent expandtab sw=2

// this is the class we want to test
#include <GateHounsfieldToMaterialsBuilder.hh>

// these classes play a role
#include <GateMaterialDatabase.hh>
#include <GateHounsfieldMaterialTable.hh>
#include <GateMessageManager.hh>

// definition of g and cm3
#include <G4SystemOfUnits.hh>

// deal with global state :-(
#include <G4Material.hh>

// standard library
#include <string>
#include <iterator> // for std::next

// preprocessor trick copied from https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html
// (with str=STRINGIZE and xstr=XSTRINGIZE)
// We could get rid of this if we would generate our own Materials.db input file.
#ifdef GATE_SOURCE_DIR
#define XSTRINGIZE(s) STRINGIZE(s)
#define STRINGIZE(s) #s
#define GATE_SOURCE_DIR_STRING XSTRINGIZE(GATE_SOURCE_DIR)
#else
#error "GATE_SOURCE_DIR not defined"
#endif


#include <catch.hpp>

GateHounsfieldMaterialTable* GetHounsfieldMaterialTableFromFile(std::string filepath,int minHU=-100000);
void write_HU_input_files(const std::string& material_path,const std::string& density_path);

TEST_CASE("Correctly define materials db and HU lookup table based on input materials and density table","[HU][density][CT][tables][GateRTion]"){
  auto G4table = G4Material::GetMaterialTable();
  size_t ng4material = G4table->size();
  // TODO: maybe use random filenames so that tests can run in parallel.
  std::string input_material_file("tmp_HUbuilder_test_input_material_file.txt");
  std::string input_density_file("tmp_HUbuilder_test_input_density_table.txt");
  std::string output_db_file("tmp_HUbuilder_test_output_db_file.txt");
  std::string output_HU_material_file("tmp_HUbuilder_test_output_HU_material_file.txt");

  // allow debugging output (maybe move this to main program?)
  GateMessageManager *messman = GateMessageManager::GetInstance();
  messman->SetMessageLevel("Geometry",1);
  messman->SetMessageLevel("Core",1);

  // start with the standard material database (needed for properties of elements)
  theMaterialDatabase.AddMDBFile( GATE_SOURCE_DIR_STRING "/GateMaterials.db" );

  // create input files
  write_HU_input_files(input_material_file,input_density_file);

  // configure HU to material table builder
  INFO( "Constructing table builder, not building any tables yet...");
  auto hu2mb = new GateHounsfieldToMaterialsBuilder;

  INFO( "Setting input and output filenames");
  hu2mb->SetMaterialTable(input_material_file);
  hu2mb->SetDensityTable(input_density_file);
  hu2mb->SetOutputMaterialDatabaseFilename(output_db_file);
  hu2mb->SetOutputHUMaterialFilename(output_HU_material_file);
  CHECK(true);

  for (double dtol = 1.0 ; dtol>0.9e-4 ; dtol/=10 ){
    // SECTION( string("Build and write with dtol=")+std::to_string(dtol)+" g/cm3");
    SECTION( std::string("Build and write with dtol=") + std::to_string(dtol) + " g/cm3"){
      INFO( "set density tolerance to dtol=" << dtol << " g/cm3" );
      hu2mb->SetDensityTolerance( dtol * g/cm3); // add unit, very important...
      INFO( "build standard Schneider table without crashing" );
      REQUIRE_NOTHROW( hu2mb->BuildAndWriteMaterials() );
    }
  }

  SECTION( "Check that output tables are consistent with input tables"){
    double dtol = 0.001;
    hu2mb->SetDensityTolerance( dtol * g/cm3); // add unit, very important...
    hu2mb->BuildAndWriteMaterials();
    GateMaterialDatabase::DeleteInstance();
    theMaterialDatabase.AddMDBFile( output_db_file );
    auto humtable = GetHounsfieldMaterialTableFromFile(output_HU_material_file);
    std::vector<int>   hu_input{  -1000, -98,     -97,  14,   23,     100,     101,    1600,3000};
    std::vector<double> d_input{1.21e-3,0.93,0.930486,1.03,1.031,1.119900,1.076200,1.964200, 2.8};
    auto hu = hu_input.begin();
    auto d = d_input.begin();
    for (;hu != hu_input.end() && d != d_input.end(); ++hu, ++d ){ 
      auto material =(*humtable)[humtable->GetLabelFromH(*hu)].mMaterial;
      // the value to test
      double d_lookup = material->GetDensity()/(g/cm3);
      // within the bins the max variation in density is dtol, the lookup value should be in the middle
      double tolerance = 0.6*dtol;
      // the schneider dip is a special case
      if (std::next(d) != d_input.end() && (*(std::next(d))<*d) ) tolerance = *d - *(std::next(d));
      if (d != d_input.begin() && (*(std::prev(d))>*d) ) tolerance = *(std::prev(d))-*d;
      INFO( "checking that for HU = " << double(*hu) << " ('" << material->GetName() << "') the density "
            << d_lookup << "g/cm3 obtained from the generated table, is close enough (within "
            << double(0.6*dtol) << " g/cm3) to the density "
            << double(*d) << " g/cm3 from the input table" );
      // now test that it is indeed close enough
      CHECK( d_lookup == Approx(*d).margin(tolerance) );
    }
  }

  INFO( "Cleanup" );
  /**/
  for (auto &s : std::vector<std::string>{input_material_file,input_density_file,output_db_file,output_HU_material_file}){
      if ( std::remove(s.c_str()) == 0 ){
          INFO( std::string("deleted: ") + s );
      } else {
          INFO( std::string("not found: ") + s );
          CHECK(false);
      }
  }
  /**/
  delete hu2mb;
  GateMaterialDatabase::DeleteInstance();

  // clean up the extra entries in the G4 material table
  for (size_t ig4 = ng4material; ig4 < G4table->size(); ++ig4){
    delete G4table->at(ig4);
  }
  G4table->resize(ng4material);
};

/******************************************************************************************************************/
// input data
void write_HU_input_files(const std::string& material_path,const std::string& density_path){
    std::ofstream fmaterials(material_path);
    // copy from GateContrib/dosimetry/Radiotherapy/example2/data/SimpleMaterialsTable.txt
    fmaterials
        << "[Elements] " << std::endl
        << "Hydrogen  Carbon Nitrogen Oxygen Sodium Magnesium Phosphor Sulfur " << std::endl
        << "Chlorine Argon Potassium Calcium " << std::endl
        << "Titanium Copper Zinc  Silver Tin  " << std::endl
        << "[/Elements]" << std::endl
        << "# ===============================================================================" << std::endl
        << "# HU      H    C    N    O   Na  Mg   P   S   Cl  Ar  K   Ca  Ti  Cu Zn  Ag  Sn" << std::endl
        << "# ===============================================================================" << std::endl
        << " -1050    0    0  75.5 23.2  0   0    0   0   0  1.3  0    0  0   0  0   0   0      Air       " << std::endl
        << "  -950  10.3 10.5  3.1 74.9 0.2  0   0.2 0.3 0.3  0  0.2   0  0   0  0   0   0      Lung" << std::endl
        << "  -120  11.6 68.1  0.2 19.8 0.1  0    0  0.1 0.1  0   0    0  0   0  0   0   0    AT_AG_SI1    " << std::endl
        << "  -82   11.3 56.7  0.9 30.8 0.1  0    0  0.1 0.1  0   0    0  0   0  0   0   0    AT_AG_SI2    " << std::endl
        << "  -52   11.0 45.8  1.5 41.1 0.1  0   0.1 0.2 0.2  0   0    0  0   0  0   0   0    AT_AG_SI3    " << std::endl
        << "  -22   10.8 35.6  2.2 50.9  0   0   0.1 0.2 0.2  0   0    0  0   0  0   0   0    AT_AG_SI4    " << std::endl
        << "   8    10.6 28.4  2.6 57.8  0   0   0.1 0.2 0.2  0  0.1   0  0   0  0   0   0    AT_AG_SI5    " << std::endl
        << "   19   10.3 13.4  3.0 72.3 0.2  0   0.2 0.2 0.2  0  0.2   0  0   0  0   0   0   SoftTissus   " << std::endl
        << "   80    9.4 20.7  6.2 62.2 0.6  0    0  0.6 0.3  0  0.0   0  0   0  0   0   0 ConnectiveTissue" << std::endl
        << "  120    9.5 45.5  2.5 35.5 0.1  0   2.1 0.1 0.1  0  0.1  4.5 0   0  0   0   0  Marrow_Bone01   " << std::endl
        << "  200    8.9 42.3  2.7 36.3 0.1  0   3.0 0.1 0.1  0  0.1  6.4 0   0  0   0   0  Marrow_Bone02   " << std::endl
        << "  300    8.2 39.1  2.9 37.2 0.1  0   3.9 0.1 0.1  0  0.1  8.3 0   0  0   0   0  Marrow_Bone03   " << std::endl
        << "  400    7.6 36.1  3.0 38.0 0.1 0.1  4.7 0.2 0.1  0   0  10.1 0   0  0   0   0  Marrow_Bone04   " << std::endl
        << "  500    7.1 33.5  3.2 38.7 0.1 0.1  5.4 0.2    0 0   0  11.7 0   0  0   0   0  Marrow_Bone05  " << std::endl
        << "  600    6.6 31.0  3.3 39.4 0.1 0.1  6.1 0.2  0   0   0  13.2 0   0  0   0   0  Marrow_Bone06   " << std::endl
        << "  700    6.1 28.7  3.5 40.0 0.1 0.1  6.7 0.2  0   0   0  14.6 0   0  0   0   0  Marrow_Bone07   " << std::endl
        << "  800    5.6 26.5  3.6 40.5 0.1 0.2  7.3 0.3  0   0   0  15.9 0   0  0   0   0   Marrow_Bone08   " << std::endl
        << "  900    5.2 24.6  3.7 41.1 0.1 0.2  7.8 0.3  0   0   0  17.0 0   0  0   0   0  Marrow_Bone09   " << std::endl
        << "  1000   4.9 22.7  3.8 41.6 0.1 0.2  8.3 0.3  0   0   0  18.1 0   0  0   0   0  Marrow_Bone10   " << std::endl
        << "  1100   4.5 21.0  3.9 42.0 0.1 0.2  8.8 0.3  0   0   0  19.2 0   0  0   0   0  Marrow_Bone11   " << std::endl
        << "  1200   4.2 19.4  4.0 42.5 0.1 0.2  9.2 0.3  0   0   0  20.1 0   0  0   0   0  Marrow_Bone12   " << std::endl
        << "  1300   3.9 17.9  4.1 42.9 0.1 0.2  9.6 0.3  0   0   0  21.0 0   0  0   0   0  Marrow_Bone13   " << std::endl
        << "  1400   3.6 16.5  4.2 43.2 0.1 0.2 10.0 0.3  0   0   0  21.9 0   0  0   0   0  Marrow_Bone14   " << std::endl
        << "  1500   3.4 15.5  4.2 43.5 0.1 0.2 10.3 0.3  0   0   0  22.5 0   0  0   0   0  Marrow_Bone15   " << std::endl
        << "  1640   0   0     0   0    0   0   0    0    0   0   0    0  0    4  2  65  29  AmalgamTooth" << std::endl
        << "  2300   0   0     0   0    0   0   0    0    0   0   0    0  100  0  0  0   0  MetallImplants" << std::endl
        << "  3000   0   0     0   0    0   0   0    0    0   0   0    0  100  0  0  0   0  MetallImplants" << std::endl
        << "" << std::endl
        << std::endl;
    fmaterials.close();
    std::ofstream fdensities(density_path);
    // copy from GateContrib/dosimetry/Radiotherapy/example2/data/Schneider2000DensitiesTable.txt
    fdensities
        << "# ===================" << std::endl
        << "# HU\tdensity g/cm3" << std::endl
        << "# ===================" << std::endl
        << "-1000\t1.21e-3" << std::endl
        << "-98\t0.93" << std::endl
        << "-97\t0.930486" << std::endl
        << "14\t1.03" << std::endl
        << "23\t1.031" << std::endl
        << "100\t1.119900" << std::endl
        << "101\t1.076200" << std::endl
        << "1600\t1.964200" << std::endl
        << "3000\t2.8" << std::endl;
    fdensities.close();
}


/******************************************************************************************************************/
// The code below is copied from the GateVImageVolume::LoadImageMaterialsFromHounsfieldTable() implementation.
// If this were method of the GateHounsfieldMaterialTable class then we would not have to copy it here...
GateHounsfieldMaterialTable* GetHounsfieldMaterialTableFromFile(std::string filepath,int minHU){
  std::ifstream is;
  OpenFileInput(filepath, is);
  auto humtable = new GateHounsfieldMaterialTable;
  while (is) {
    skipComment(is);
    double h1,h2;
    G4String n;
    is >> h1;
    is >> h2;
    is >> n;

    if (is) {
      if (h2 >= minHU) {
        if (h1 < minHU)
          h1 = minHU;
        humtable->AddMaterial(h1,h2,n);
      }
    }
  }
  return humtable;
}
