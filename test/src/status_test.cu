#include "catch.hpp"
#include "base.cuh"

TEST_CASE("Status", "[status]")
{
    using namespace warpcore;

    SECTION("general")
    {
        Status s = Status();
        CHECK(s == Status::none());

        s += Status::unknown_error();
        CHECK(s == Status::unknown_error());
        CHECK(s.has_unknown_error());
        CHECK(s.has_any());
        CHECK(!Status::none());

        s = s + Status::key_not_found();
        CHECK(s.has_key_not_found());
        CHECK(s.has_unknown_error());
        CHECK(s.has_any());
        CHECK(s.has_any(Status::unknown_error()));
        CHECK(!Status::none());
        CHECK(!s.has_all(~Status::invalid_key()));
        CHECK(s.has_all(Status::key_not_found() + Status::unknown_error()));

        s -= Status::unknown_error();
        CHECK(!s.has_unknown_error());
        CHECK(s.has_key_not_found());

        s = s - Status::key_not_found();
        CHECK(s == Status::none());
        CHECK((~Status::none()) != s);

        Status err = Status::error_mask();
        CHECK(err.has_any_errors());
        CHECK(!err.has_any_warnings());

        Status warn = Status::warning_mask();
        CHECK(warn.has_any_warnings());
        CHECK(!warn.has_any_errors());

        CHECK(!err.has_any(warn));
    }

    SECTION("CUDA atomics")
    {
        //TODO
    }

}