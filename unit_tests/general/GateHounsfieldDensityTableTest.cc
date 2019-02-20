#include <GateHounsfieldDensityTable.hh>
#include <G4SystemOfUnits.hh>
#include <catch.hpp>
#include <fstream>
#include <vector>
#include <utility>
#include <string>

TEST_CASE("Density table reads input correctly and can interpolate","[HU][density][CT][interpolation][GateRTion]"){

  std::string dfilename("tmp_hu_density.txt");
  std::vector<std::string> separators{" ","  ","\t","\t "," \t"," \t "};

  for (auto sep : separators){
    SECTION(std::string("test density table with sep='") + sep + std::string("' in input file") ){

      // generate input file
      std::string ws; // "random" initial/final whitespace
      std::ofstream dfile(dfilename);
      std::vector< std::pair<int,double> > hud_input{{-1000,0},{0,1.},{1000,3.}};
      for ( auto hud_value : hud_input ){
        dfile << ws << hud_value.first << sep << hud_value.second << ws << std::endl;
        ws += " \t";
      }
      dfile.close();

      // create density table
      GateHounsfieldDensityTable dtable;
      dtable.Read(dfilename);

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

// vim: et:sw=2:ai:smartindent
