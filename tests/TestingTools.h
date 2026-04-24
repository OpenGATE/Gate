/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About class: Testing utility header providing the CHECK macro for assertion-style test failure reporting with function name and line number; used by all positronium unit tests.
 **/

#ifndef TestingTools_h
#define TestingTools_h


// ---- Test helper ----
#define CHECK(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cerr <<  "Test failure: " << msg \
                << " (in " << __FUNCTION__ << ", line " << __LINE__ << ")" <<  "\n"; \
      return false; \
    } \
  } while(0)

#endif
