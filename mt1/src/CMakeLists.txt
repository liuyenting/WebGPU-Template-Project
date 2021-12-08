add_executable(
    mt1
    main.cu
)

target_link_libraries(
    mt1
    PRIVATE
        libwb
)
