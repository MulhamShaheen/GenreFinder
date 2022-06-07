import * as FilePond from 'filepond'
// Get a file input reference
console.log(1)

const input = document.getElementById('file');
// Create a FilePond instance
create(input, {
    storeAsFile: true,
});
const pond = FilePond.create(inputElement);