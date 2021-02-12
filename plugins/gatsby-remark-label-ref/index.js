const visit = require("unist-util-visit")
const toString = require("mdast-util-to-string")
const label_regex = /\\label{([^}]+?)}/g;
const href_regex = /\\href{([^}]+?)}/g;
const ref_regex = /\\ref{([^}]+?)}/g;

function replaceAll(labelRegistry, text, regex, templateFn) {
    let matches = Array.from(text.matchAll(regex));
    if (matches.length < 1) return text;
    matches.forEach(match => {
        let id = match[1];
        let parts = id.split(':');
        let idType = parts[0];
        let idName = parts[1];        
        if ((idType in labelRegistry) &&
            (idName in labelRegistry[idType])) {
            
            let count = labelRegistry[idType][idName];
            const html = templateFn(id, count);
            text = text.replace(match[0], html);
        }
    });
    return text
}

module.exports = ({ markdownAST, markdownNode }) => {    
    labelRegistry = {};    
    let allLabelMatches = Array.from(
        markdownNode.internal.content.matchAll(label_regex));
    
    allLabelMatches.forEach((match) => {
        let id = match[1];
        let parts = id.split(':');
        let idType = parts[0];
        let idName = parts[1];
        if (!(idType in labelRegistry)) {
            labelRegistry[idType] = {
                '__count__': 0
            }
        }
        if (idName in labelRegistry[idType]) {
            throw Error(`\\label{${id}} was defined twice.`);
        }
        labelRegistry[idType][idName] = ++labelRegistry[idType]['__count__'];
    });

    visit(markdownAST, "text", node => {
        let text = toString(node);
        let matches = Array.from(text.matchAll(label_regex));
        if (matches.length < 1) return;
        if (matches.length > 1) {
            throw Error('Expect only one \\label{} per line.');
        }
        let id = matches[0][1];
        let parts = id.split(':');
        let idType = parts[0];
        let idName = parts[1];

        if (!(idType in labelRegistry) || !(idName in labelRegistry[idType])) {
            throw Error(`Something when wrong. ${matches[0][0]} not found.`);
        }
        const html = `<p><span id="${id}"></span></p>`;
        node.type = "html";        
        node.value = html;
    });

    visit(markdownAST, ["math", "text"], node => {        
        let text = toString(node);
        newText = replaceAll(labelRegistry, text, href_regex,
            (id, count) => `<a href="#${id}">${count}</a>`);
        newText = replaceAll(labelRegistry, newText, ref_regex, (id, count) => `${count}`);
        if (text != newText) {
            if (node.type == "text") {
                node.type = "html";
            }
            node.value = newText;
        }
    });

    return markdownAST
}