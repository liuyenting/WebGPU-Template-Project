add_executable(
    mp1
    main.cu
)

target_link_libraries(
    mp1
    PRIVATE
        libwb
)
