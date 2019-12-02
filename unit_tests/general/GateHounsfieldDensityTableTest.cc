#include <GateHounsfieldDensityTable.hh>
#include <G4SystemOfUnits.hh>
#include <catch.hpp>
#include <fstream>
#include <vector>
#include <utility>
#include <string>
#include <sstream>

TEST_CASE("Density table reads input correctly and can interpolate","[HU][density][CT][interpolation][GateRTion]"){
// todo; suggestion, one could do two different tests here, one that tests that input is read correctly
// for different separators, and a second test that tests that it interpolates the data correctly. In that way each test
// tests a single piece of functionality

  std::vector<std::string> separators{" ","  ","\t","\t "," \t"," \t "};

  for (auto sep : separators){
    SECTION(std::string("test density table with sep='") + sep + std::string("' in input file") ){

      // generate input data
      std::stringstream in_stream;

      std::string ws; // "random" initial/final whitespace
      std::vector< std::pair<int,double> > hud_input{{-1000,0},{0,1.},{1000,3.}};
      for ( auto hud_value : hud_input ){
        in_stream << ws << hud_value.first << sep << hud_value.second << ws << std::endl;
        ws += " \t";
      }

      // create density table
      GateHounsfieldDensityTable dtable;
      dtable.ReadFromStream(in_stream, G4String("unittest stream"));

      // reproduce input values
      for ( auto hud_value : hud_input ){
        INFO("test that HU=" << hud_value.first << " gives exactly input density D=" << hud_value.second );
        REQUIRE( dtable.GetDensityFromH( hud_value.first)/(g/cm3) == hud_value.second );
      }

      // interpolation
      for ( int hu = 1; hu<1000; ++hu ){
        double d_int = 1. + hu * 0.002;
        INFO("test that HU=" << hu << " gives exactly the interpolated density D=" << d_int );
        CHECK( dtable.GetDensityFromH(+hu)/(g/cm3) == Approx(d_int).margin(1e-10) );
        d_int = 1. - hu * 0.001;
        CHECK( dtable.GetDensityFromH(-hu)/(g/cm3) == Approx(d_int).margin(1e-10) );
      }

      // out of range
      INFO("test that out of range HU=" << -1001 << " gives exactly the minimum input density D=" << 0. );
      CHECK( dtable.GetDensityFromH(-1001)/(g/cm3) == 0. );
      INFO("test that out of range HU=" << -2001 << " gives exactly the minimum input density D=" << 0. );
      CHECK( dtable.GetDensityFromH(-2001)/(g/cm3) == 0. );
      INFO("test that out of range HU=" << +1001 << " gives exactly the maxnimum input density D=" << 3. );
      CHECK( dtable.GetDensityFromH(+1001)/(g/cm3) == 3. );
      INFO("test that out of range HU=" << +2001 << " gives exactly the maximum input density D=" << 3. );
      CHECK( dtable.GetDensityFromH(+2001)/(g/cm3) == 3. );
    }
  }
}

TEST_CASE("GateHounsfield table can deal with comments in input data","[HU][density][CT][interpolation][GateRTion]"){

  std::stringstream in_stream;
  in_stream << "# ======== " << std::endl
            << "-1000 1.21e-3" << std::endl
            << "# another comment" << std::endl
            << "0 1" << std::endl
            << "# final comment" << std::endl;

  GateHounsfieldDensityTable dtable;
  dtable.ReadFromStream(in_stream, G4String("unittest stream"));

  REQUIRE(dtable.GetDensityFromH(-1000)/(g/cm3) == Approx(1.21e-3).epsilon(0.001));
  REQUIRE(dtable.GetDensityFromH(0)/(g/cm3) == Approx(1.0).epsilon(0.001));
}

TEST_CASE("GateHounsfield table throws exception if input data is not ordered","[HU][density][CT][interpolation][GateRTion]"){

  std::stringstream in_stream;
  in_stream << "-1000 1.21e-3" << std::endl
            << "0 1" << std::endl
            << "-100 0.1" << std::endl;

  GateHounsfieldDensityTable dtable;
  // REQUIRE_THROWS(dtable.ReadFromStream(in_stream, G4String("unittest stream")));
  // TODO: Implement a way to catch GateErrors
}

// vim: et:sw=2:ai:smartindent
