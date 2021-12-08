add_executable(
    mp2
    main.cu
)

target_link_libraries(
    mp2
    PRIVATE
        libwb
)
