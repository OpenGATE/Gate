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
